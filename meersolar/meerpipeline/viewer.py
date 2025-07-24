import os
import platform
import ctypes
import sys
import numpy as np
import traceback
import argparse
import webbrowser
from collections import deque
from threading import Thread
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QSplitter,
    QSizeGrip,
    QGridLayout,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

LOG_DIR = None
POSIX_FADV_DONTNEED = 4
libc = ctypes.CDLL("libc.so.6")


#####################################
# Resource management
#####################################
def drop_file_cache(filepath, verbose=False):
    """
    Advise the OS to drop the given file from the page cache.
    Safe, per-file, no sudo required.
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    try:
        if not os.path.isfile(filepath):
            return
        fd = os.open(filepath, os.O_RDONLY)
        result = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        os.close(fd)
        if verbose:
            if result == 0:
                print(f"[cache drop] Released: {filepath}")
            else:
                print(f"[cache drop] Failed ({result}) for: {filepath}")
    except Exception as e:
        if verbose:
            print(f"[cache drop] Error for {filepath}: {e}")
            traceback.print_exc()


def drop_cache(path, verbose=False):
    """
    Drop file cache for a file or all files under a directory.

    Parameters
    ----------
    path : str
        File or directory path
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    if os.path.isfile(path):
        drop_file_cache(path, verbose=verbose)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                full_path = os.path.join(root, f)
                drop_file_cache(full_path, verbose=verbose)
    else:
        if verbose:
            print(f"[cache drop] Path does not exist or is not valid: {path}")


def get_cachedir():
    homedir = os.environ.get("HOME")
    if homedir is None:
        homedir = os.path.expanduser("~")
    username = os.getlogin()
    cachedir = f"{homedir}/.solarpipe"
    os.makedirs(cachedir, exist_ok=True)
    return cachedir


class SmartDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        # Don't show default for boolean flags
        if isinstance(action, argparse._StoreTrueAction) or isinstance(
            action, argparse._StoreFalseAction
        ):
            return action.help
        return super()._get_help_string(action)


def get_logid(logfile):
    """
    Get log id for remote logger from logfile name
    """
    name = os.path.basename(logfile)
    logmap = {
        "apply_basiccal.log": "Applying basic calibration solutions",
        "apply_pbcor.log": "Applying primary beam corrections",
        "apply_selfcal.log": "Applying self-calibration solutions",
        "basic_cal.log": "Basic calibration",
        "cor_sidereal_selfcals.log": "Correction of sidereal motion before self-calibration",
        "cor_sidereal_targets.log": "Correction of sidereal motion for target scans",
        "flagging_cal_calibrator.log": "Basic flagging",
        "modeling_calibrator.log": "Simulating visibilities of calibrators",
        "split_targets.log": "Spliting target scans",
        "split_selfcals.log": "Spliting for self-calibration",
        "selfcal_targets.log": "All self-calibrations",
        "imaging_targets.log": "All imaging",
        "noise_cal.log": "Flux calibration using noise-diode",
        "partition_cal.log": "Partioning for basic calibration",
        "ds_targets.log": "Making dynamic spectra",
        "main.log":"Main pipeline logs",
    }

    if name in logmap:
        return logmap[name]
    elif "selfcals_scan_" in name:
        name = name.rstrip("_selfcal.log")
        scan = name.split("scan_")[-1].split("_spw")[0]
        spw = name.split("spw_")[-1].split("_selfcal")[0]
        return f"Self-calibration for: Scan : {scan}, Spectral window: {spw}"
    elif "imaging_targets_scan_" in name:
        name = name.rstrip(".log")
        scan = name.split("scan_")[-1].split("_spw")[0]
        spw = name.split("spw_")[-1].split("_selfcal")[0]
        return f"Imaging for: Scan : {scan}, Spectral window: {spw}"
    else:
        return name


class TailWatcher(FileSystemEventHandler, QObject):
    new_line = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._running = True
        self._position = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.observer = Observer()

    def start(self):
        self.observer.schedule(
            self, path=os.path.dirname(self.file_path), recursive=False
        )
        self.observer.start()
        # Do not emit initial content to avoid duplication

    def stop(self):
        self._running = False
        self.observer.stop()
        self.observer.join()

    def on_modified(self, event):
        if event.src_path == self.file_path and self._running:
            try:
                with open(self.file_path, "r") as f:
                    f.seek(self._position)
                    new_data = f.read()
                    self._position = f.tell()
                    if new_data:
                        # Remove blank/whitespace-only lines
                        filtered_lines = "\n".join(
                            line for line in new_data.splitlines() if line.strip()
                        )
                        if filtered_lines:
                            self.new_line.emit(filtered_lines + "\n")
            except Exception as e:
                self.new_line.emit(f"\n[watcher error] {e}\n")


class LogViewer(QWidget):
    def __init__(self, max_lines=10000):
        super().__init__()
        self.setWindowTitle("MeerLogger (Live Log)")
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(
            screen.x() + screen.width() // 10,
            screen.y() + screen.height() // 10,
            int(screen.width() * 0.8),
            int(screen.height() * 0.8),
        )

        self.max_lines = max_lines
        self.buffer = []
        self.tail_watcher = None
        self.current_log_path = None

        self.setup_ui()
        self.refresh_logs()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_logs)
        self.refresh_timer.start(2000)

    def calc_list_width(self):
        fm = self.log_list.fontMetrics()
        widths = [
            fm.horizontalAdvance(self.log_list.item(i).text())
            for i in range(self.log_list.count())
        ]
        return max(150, min(int(1.1 * max(widths, default=100)), 500))

    def setup_ui(self):
        outer_layout = QGridLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        inner_layout = QVBoxLayout()
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        self.log_list = QListWidget()
        self.log_list.itemClicked.connect(self.load_log_content)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        self.splitter.addWidget(self.log_list)
        self.splitter.addWidget(self.log_view)
        inner_layout.addWidget(self.splitter)

        grip_layout = QHBoxLayout()
        grip_layout.addStretch()
        grip_layout.addWidget(QSizeGrip(self))
        inner_layout.addLayout(grip_layout)

        inner_container = QWidget()
        inner_container.setObjectName("InnerContainer")
        inner_container.setLayout(inner_layout)
        inner_container.setStyleSheet(
            """
            QWidget#InnerContainer {
                background-color: #f0f0f0;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
            """
        )

        outer_layout.addWidget(inner_container, 0, 0)

    def refresh_logs(self):
        existing_paths = {
            self.log_list.item(i).data(Qt.UserRole)
            for i in range(self.log_list.count())
        }
        new_items_added = False

        if os.path.isdir(LOG_DIR):
            log_files = [
                fname
                for fname in os.listdir(LOG_DIR)
                if os.path.isfile(os.path.join(LOG_DIR, fname))
                and fname.endswith(".log")
            ]
            log_files.sort(key=lambda f: os.path.getctime(os.path.join(LOG_DIR, f)))

            for fname in log_files:
                full_path = os.path.join(LOG_DIR, fname)
                if full_path not in existing_paths:
                    display_name = get_logid(fname)
                    item = QListWidgetItem(display_name)
                    item.setData(Qt.UserRole, full_path)
                    self.log_list.addItem(item)
                    new_items_added = True

        if new_items_added:
            QTimer.singleShot(
                100,
                lambda: self.splitter.setSizes(
                    [self.calc_list_width(), self.width() - self.calc_list_width()]
                ),
            )

    def load_log_content(self, item):
        new_log_path = item.data(Qt.UserRole)
        if self.tail_watcher:
            self.tail_watcher.stop()
        self.current_log_path = new_log_path
        self.buffer.clear()
        self.log_view.clear()

        try:
            with open(new_log_path, "r") as f:
                full_data = f.read()
                # Split, filter blank lines, and rejoin with original line
                # endings
                lines = [
                    line for line in full_data.splitlines(keepends=True) if line.strip()
                ]
                self.buffer = lines
                self.log_view.setPlainText("".join(lines))
                self.log_view.moveCursor(QTextCursor.End)
        except Exception as e:
            self.buffer = [f"[Error reading file: {e}]\n"]
            self.log_view.setPlainText(self.buffer[0])

        self.tail_watcher = TailWatcher(self.current_log_path)
        self.tail_watcher.new_line.connect(self.append_log_line)
        self.tail_watcher.start()

    def append_log_line(self, text):
        lines = text.splitlines(keepends=True)
        self.buffer.extend(lines)
        if len(self.buffer) > self.max_lines:
            self.buffer = self.buffer[-self.max_lines :]
        self.log_view.setPlainText("".join(self.buffer))
        self.log_view.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        if self.tail_watcher:
            self.tail_watcher.stop()
        QApplication.quit()


def cli():
    global LOG_DIR
    parser = argparse.ArgumentParser(
        description="MeerSOLAR Logger", formatter_class=SmartDefaultsHelpFormatter
    )
    parser.add_argument("--jobid", type=str, default=None, help="MeerSOLAR Job ID")
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Name of log directory",
    )
    parser.add_argument(
        "--no-prefect",
        action="store_false",
        dest="prefect",
        help="Name of log directory",
    )
    args = parser.parse_args()

    if args.prefect:
        cachedir = get_cachedir()
        if os.path.exists(f"{cachedir}/prefect.dashboard"):
            with open(f"{cachedir}/prefect.dashboard","r") as f:
                SERVER_DASHBOARD=f.read()
            webbrowser.open(SERVER_DASHBOARD)
            sys.exit(0)
        else:
            if args.jobid is None and args.logdir is None:
                print("Please provide either job ID or log directory.")
                sys.exit(1)
    else:
        if args.jobid is not None:
            jobfile_name = f"{cachedir}/main_pids_{args.jobid}.txt"
            if not os.path.exists(jobfile_name):
                print(
                    f"Job ID: {args.jobid} is not available. Provide log directory name."
                )
                sys.exit(1)
            else:
                results = np.loadtxt(jobfile_name, dtype="str", unpack=True)
                workdir = str(results[3])
                if not os.path.exists(workdir):
                    print(f"Work directory : {workdir} is not present.")
                    sys.exit(1)
                LOG_DIR = workdir.rstrip("/") + "/logs"
        else:                       
            if not os.path.exists(args.logdir):
                print(
                    f"Log directory: {args.logdir} is not present. Please provide a valid log directory."
                )
                sys.exit(1)
            LOG_DIR = args.logdir

    os.environ["QT_OPENGL"] = "software"
    os.environ["QT_XCB_GL_INTEGRATION"] = "none"
    os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
    os.makedirs(f"{LOG_DIR}/xdgtmp", exist_ok=True)
    os.chmod(f"{LOG_DIR}/xdgtmp", 0o700)
    os.environ.setdefault("XDG_RUNTIME_DIR", f"{LOG_DIR}/xdgtmp")
    os.environ["XDG_RUNTIME_DIR"] = f"{LOG_DIR}/xdgtmp"
    os.environ["TMPDIR"] = f"{LOG_DIR}/xdgtmp"

    try:
        app = QApplication(sys.argv)
        app.setStyleSheet(
            """
            * {
                font-family: \"Segoe UI\", \"Noto Sans\", \"Sans Serif\";
                font-size: 15px;
            }
            QListWidget, QPlainTextEdit, QPushButton {
                font-size: 15px;
            }
            QPushButton {
                padding: 4px 14px;
            }
        """
        )
        viewer = LogViewer()
        viewer.show()
        sys.exit(app.exec_())
    except Exception as e:
        traceback.print_exc()
    finally:
        if LOG_DIR is not None and os.path.exists(LOG_DIR):
            drop_cache(LOG_DIR)
