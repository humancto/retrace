//! Log to `<root>/logs/retrace.log.<date>` (daily rolling, the newest 14 files are kept), plus
//! stderr in debug builds or when `RETRACE_LOG_STDERR=1`.

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

/// Days of log files to keep in `<root>/logs`.
const MAX_LOG_FILES: usize = 14;

/// Keep the guard alive for the life of the process, or buffered lines are lost.
pub fn init(logs: &std::path::Path) -> WorkerGuard {
    use tracing_appender::rolling::{RollingFileAppender, Rotation};
    let file = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix("retrace.log")
        .max_log_files(MAX_LOG_FILES)
        .build(logs)
        .unwrap_or_else(|e| {
            eprintln!("retrace: log rotation setup failed ({e}); old logs are not deleted");
            tracing_appender::rolling::daily(logs, "retrace.log")
        });
    let (writer, guard) = tracing_appender::non_blocking(file);
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=warn,usls=warn,tao=warn,wry=warn"));
    let to_stderr = cfg!(debug_assertions) || std::env::var_os("RETRACE_LOG_STDERR").is_some();
    let stderr = to_stderr.then(|| fmt::layer().with_writer(std::io::stderr));
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_ansi(false).with_writer(writer))
        .with(stderr)
        .try_init();
    guard
}
