import os
import tqdm
import logging


def get_logger(
        name: str = "exp",
        log_level: int = logging.INFO,
        log_file: str = None,
        log_file_mode: str = "w",
        is_main_process: bool = True,
):
    """Get a logger that prints to both console and file.

    Args:
        name: The name of the logger.
        log_level: The logging level. Note that levels of non-main processes are always "ERROR".
        log_file: The path to the log file. If None, the log file is disabled.
        log_file_mode: The mode to open the log file.
        is_main_process: Whether the logger is for the main process.
    """
    logger = logging.getLogger(name)
    # check if the logger exists
    if logger.hasHandlers():
        return logger
    # add a stream handler
    handlers: list = [TqdmLoggingHandler()]
    # add a file handler for main process
    if is_main_process and log_file is not None:
        file_handler = logging.FileHandler(log_file, log_file_mode)
        handlers.append(file_handler)
    # set format & level for all handlers
    # note that levels of non-main processes are always "ERROR"
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_level = log_level if is_main_process else logging.ERROR
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that uses tqdm.write() to print log messages.

    This handler prevents the logging messages from interfering with tqdm progress bars. Note that
    you need to use `import tqdm` instead of `from tqdm import tqdm` to make this handler work.

    References:
      - https://stackoverflow.com/a/38739634/23025233
    """
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:  # noqa
            self.handleError(record)


class StatusTracker:
    """Track status and print to logger and tensorboard."""
    def __init__(
            self,
            logger: logging.Logger,
            print_freq: int,
            tensorboard_dir: str = None,
            is_main_process: bool = True,
    ):
        self.logger = logger
        self.print_freq = print_freq

        self.tb_writer = None
        if is_main_process and tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(tensorboard_dir)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()

    def track_status(self, name: str, status: dict, step: int):
        # logger
        if self.print_freq > 0 and (step + 1) % self.print_freq == 0:
            message = f"[{name}] step: {step}" + "".join(f", {k}: {v:.6f}" for k, v in status.items())
            self.logger.info(message)
        # tensorboard
        for i, (k, v) in enumerate(status.items()):
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f"{name}/{k}", v, step)
