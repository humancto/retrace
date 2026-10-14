//! Enrichment path: a VLM (llama-server, mlx-vlm, Ollama… any OpenAI-compatible endpoint) turns a
//! tracklet's best crop + context frame into a fixed JSON caption, which feeds FTS and the UI.

use crate::db::Db;
use anyhow::{Context, Result};
use base64::Engine;
use serde::Deserialize;
use std::path::Path;
use std::time::Duration;

pub const CAPTION_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "class": {"type": "string", "enum": ["person", "vehicle", "animal", "package", "unknown"]},
    "upper_color": {"type": "string", "enum": ["black","white","gray","red","blue","green","yellow","brown","other","unknown"]},
    "lower_color": {"type": "string", "enum": ["black","white","gray","red","blue","green","yellow","brown","other","unknown"]},
    "headwear": {"type": "string", "enum": ["none","cap","hood","helmet","hat","unknown"]},
    "carried_objects": {"type": "array", "items": {"type": "string", "enum": ["backpack","bag","box","package","ladder","tool","phone","umbrella","stroller","other"]}},
    "vehicle_type": {"type": "string", "enum": ["none","sedan","suv","pickup","van","truck","motorcycle","bicycle","bus","unknown"]},
    "vehicle_color": {"type": "string", "enum": ["none","black","white","gray","silver","red","blue","green","yellow","brown","other","unknown"]},
    "action": {"type": "string", "enum": ["walking","running","standing","loitering","sitting","entering","exiting","carrying","placing","picking_up","opening_door","driving","parking","reversing","falling","climbing","unknown"]},
    "anomaly": {"type": "string"},
    "color_confidence": {"type": "string", "enum": ["high","low","none_ir"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["class","upper_color","lower_color","headwear","carried_objects","vehicle_type","vehicle_color","action","anomaly","color_confidence","confidence"]
}"#;

#[derive(Debug, Clone)]
pub struct AnnotateOpts {
    pub endpoint: String, // e.g. http://127.0.0.1:8081/v1
    pub model: String,
    pub api_key: Option<String>,
    pub limit: usize,
    pub once: bool,
    pub timeout_s: u64,
}

#[derive(Deserialize)]
struct ChatResp {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: Msg,
}
#[derive(Deserialize)]
struct Msg {
    content: serde_json::Value,
}

fn data_url(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

pub fn caption_text(c: &serde_json::Value) -> String {
    let s = |k: &str| c[k].as_str().unwrap_or("unknown").to_string();
    let mut parts = vec![s("class")];
    if s("class") == "vehicle" {
        parts.push(format!("{} {}", s("vehicle_color"), s("vehicle_type")));
    } else {
        if s("color_confidence") != "none_ir" {
            parts.push(format!("{} top", s("upper_color")));
            parts.push(format!("{} bottom", s("lower_color")));
        }
        if s("headwear") != "none" && s("headwear") != "unknown" {
            parts.push(s("headwear"));
        }
        if let Some(arr) = c["carried_objects"].as_array() {
            for o in arr {
                if let Some(o) = o.as_str() {
                    parts.push(format!("carrying {o}"));
                }
            }
        }
    }
    parts.push(s("action"));
    if let Some(a) = c["anomaly"].as_str()
        && !a.is_empty()
        && a != "none"
    {
        parts.push(format!("anomaly: {a}"));
    }
    parts.join(", ")
}

pub async fn run_annotate(db_path: &Path, crops_dir: &Path, o: &AnnotateOpts) -> Result<usize> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(o.timeout_s))
        .build()?;
    let schema: serde_json::Value = serde_json::from_str(CAPTION_SCHEMA)?;
    let mut done = 0usize;
    loop {
        let todo = { Db::open(db_path)?.uncaptioned(o.limit)? };
        if todo.is_empty() {
            if o.once {
                break;
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
            continue;
        }
        for t in todo {
            let crops: Vec<serde_json::Value> =
                serde_json::from_str(&t.best_crops_json).unwrap_or_default();
            let Some(crop_rel) = crops.first().and_then(|c| c["path"].as_str()) else {
                continue;
            };
            let crop = data_url(&crops_dir.join(crop_rel))?;
            let ctx_rel = crop_rel
                .rsplit_once('-')
                .map(|(a, _)| format!("{a}-ctx.jpg"))
                .unwrap_or_default();
            let mut content = vec![
                serde_json::json!({"type": "text", "text": format!(
                    "You annotate security-camera footage. The first image is a crop of a tracked {} seen on camera '{}' for {:.0} seconds moving {}. The second image, if present, is the full frame for context. Describe only what is visible. If the frame is monochrome (infrared), set color_confidence to none_ir and colors to unknown. Answer with JSON only.",
                    t.class, t.cam_name, t.dwell_s, t.direction.replace('_', " "))}),
                serde_json::json!({"type": "image_url", "image_url": {"url": crop}}),
            ];
            if let Ok(ctx) = data_url(&crops_dir.join(&ctx_rel)) {
                content.push(serde_json::json!({"type": "image_url", "image_url": {"url": ctx}}));
            }
            let body = serde_json::json!({
                "model": o.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_tokens": 300,
                "response_format": {"type": "json_schema", "json_schema": {"name": "caption", "schema": schema}}
            });
            let mut req = client
                .post(format!(
                    "{}/chat/completions",
                    o.endpoint.trim_end_matches('/')
                ))
                .json(&body);
            if let Some(k) = &o.api_key {
                req = req.bearer_auth(k);
            }
            let t0 = std::time::Instant::now();
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("vlm request failed: {e}");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            };
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                tracing::warn!(
                    "vlm {status}: {}",
                    text.chars().take(300).collect::<String>()
                );
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
            let parsed: ChatResp = match serde_json::from_str(&text) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("vlm bad response: {e}");
                    continue;
                }
            };
            let content = parsed
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or(serde_json::Value::Null);
            let content_str = match &content {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let json_start = content_str.find('{').unwrap_or(0);
            let caption: serde_json::Value = match serde_json::from_str(&content_str[json_start..])
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "vlm output is not JSON ({e}): {}",
                        content_str.chars().take(200).collect::<String>()
                    );
                    continue;
                }
            };
            let txt = caption_text(&caption);
            Db::open(db_path)?.set_caption(t.id, &caption.to_string(), &txt)?;
            done += 1;
            tracing::info!(
                tracklet = t.id,
                ms = t0.elapsed().as_millis(),
                "caption: {txt}"
            );
        }
        if o.once {
            break;
        }
    }
    Ok(done)
}
