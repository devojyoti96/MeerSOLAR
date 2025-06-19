import os
import sys, numpy as np
import traceback
from optparse import OptionParser
from collections import deque
from threading import Thread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QPlainTextEdit, QSplitter, QSizeGrip, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

LOG_DIR = None  # Will be set in `main()`


def get_logid(logfile):
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
        "split_targets.log": "Splitting target scans",
        "split_selfcals.log": "Splitting for self-calibration",
        "selfcal_targets.mainlog": "All self-calibrations main log",
        "imaging_targets.mainlog": "All imaging main log",
        "selfcal_targets.log": "All self-calibrations",
        "imaging_targets.log": "All imaging",
        "noise_cal.log": "Flux calibration using noise-diode",
        "partition_cal.log": "Partitioning for basic calibration"
    }
    if name in logmap:
        return logmap[name]
    elif "selfcals_scan_" in name:
        name = name.rstrip("_selfcal.log")
        scan = name.split("scan_")[-1].split("_spw")[0]
        spw = name.split("spw_")[-1].split("_selfcal")[0]
        return f"Self-calibration for: Scan {scan}, SPW {spw}"
    elif "imaging_targets_scan_" in name:
        name = name.rstrip(".log")
        scan = name.split("scan_")[-1].split("_spw")[0]
        spw = name.split("spw_")[-1].split("_selfcal")[0]
        return f"Imaging for: Scan {scan}, SPW {spw}"
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
        self.observer.schedule(self, path=os.path.dirname(self.file_path), recursive=False)
        self.observer.start()
        Thread(target=self._emit_existing_lines, daemon=True).start()

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
                        self.new_line.emit(new_data)
            except Exception as e:
                self.new_line.emit(f"\n[watcher error] {e}\n")

    def _emit_existing_lines(self):
        try:
            with open(self.file_path, "r") as f:
                f.seek(max(0, os.path.getsize(self.file_path) - 2048))
                lines = f.read()
                self.new_line.emit(lines)
        except Exception:
            pass


class LogViewer(QWidget):
    def __init__(self, max_lines=10000):
        super().__init__()
        self.setWindowTitle("MeerLogger (Live Log)")
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen.x() + screen.width()//10, screen.y() + screen.height()//10,
                         int(screen.width() * 0.8), int(screen.height() * 0.8))

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
        widths = [fm.horizontalAdvance(self.log_list.item(i).text()) for i in range(self.log_list.count())]
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
        inner_container.setStyleSheet("""
            QWidget#InnerContainer {
                background-color: #f0f0f0;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
        """)

        outer_layout.addWidget(inner_container, 0, 0)

    def refresh_logs(self):
        existing_paths = {
            self.log_list.item(i).data(Qt.UserRole)
            for i in range(self.log_list.count())
        }

        new_items_added = False
        if os.path.isdir(LOG_DIR):
            log_files = [
                fname for fname in os.listdir(LOG_DIR)
                if os.path.isfile(os.path.join(LOG_DIR, fname)) and fname.endswith(".log")
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
            QTimer.singleShot(100, lambda: self.splitter.setSizes([
                self.calc_list_width(),
                self.width() - self.calc_list_width()
            ]))


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
                self.buffer = full_data.splitlines(keepends=True)
                self.log_view.setPlainText(full_data)
                self.log_view.moveCursor(QTextCursor.End)
        except Exception as e:
            self.buffer = [f"[Error reading file: {e}]\n"]
            self.log_view.setPlainText(self.buffer[0])

        # Then start tailing
        self.tail_watcher = TailWatcher(self.current_log_path)
        self.tail_watcher.new_line.connect(self.append_log_line)
        self.tail_watcher.start()


    def append_log_line(self, text):
        lines = text.splitlines(keepends=True)
        self.buffer.extend(lines)
        self.log_view.setPlainText("".join(self.buffer))
        self.log_view.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        if self.tail_watcher:
            self.tail_watcher.stop()
        QApplication.quit()

def get_datadir():
    """
    Get package data directory
    """
    from importlib.resources import files
    datadir_path = str(files("meersolar").joinpath("data"))
    os.makedirs(datadir_path,exist_ok=True)
    os.makedirs(f"{datadir_path}/pids",exist_ok=True)
    return datadir_path
    
def main():
    global LOG_DIR
    usage = "MeerSOLAR Logger"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--jobid",
        dest="jobid",
        default=None,
        help="MeerSOLAR Job ID",
        metavar="Integer",
    )
    parser.add_option(
        "--logdir",
        dest="logdir",
        default=None,
        help="Name of log directory",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if options.jobid==None and options.logdir==None:
        print ("Please provide either job ID or log directory.")
        sys.exit(1)
    else:   
        datadir=get_datadir()
        if options.jobid!=None:
            jobfile_name=datadir + f"/main_pids_{options.jobid}.txt"
            if os.path.exists(jobfile_name)==False:
                print (f"Job ID: {options.jobid} is not available. Provide log directory name.")
                sys.exit(1)
            else:
                results=np.loadtxt(jobfile_name,dtype="str",unpack=True)
                basedir=results[2]
                if os.path.exists(basedir)==False:
                    print ("Base directory : {basedir} is not present.")
                    sys.exit(1)
                LOG_DIR=basedir.rstrip("/")+"/logs"
        else:
            if os.path.exists(options.logdir)==False:
                print (f"Log diretory: {options.logdir} is not present. Please provide a valid log directory.")
                sys.exit(1)
            LOG_DIR=options.logdir

    # Environment fixes (must be set before QApplication loads Qt backend)
    os.environ["QT_OPENGL"] = "software"
    os.environ["QT_XCB_GL_INTEGRATION"] = "none"
    os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
    os.makedirs(f"{LOG_DIR}/xdgtmp", exist_ok=True)
    os.chmod(f"{LOG_DIR}/xdgtmp", 0o700)
    os.environ.setdefault("XDG_RUNTIME_DIR", f"{LOG_DIR}/xdgtmp")
    os.environ["XDG_RUNTIME_DIR"]= f"{LOG_DIR}/xdgtmp"
    os.environ["TMPDIR"]=f"{LOG_DIR}/xdgtmp"

    app = QApplication(sys.argv)
    app.setStyleSheet("""
    * {
        font-family: "Segoe UI", "Noto Sans", "Sans Serif";
        font-size: 15px;
    }
    QListWidget, QPlainTextEdit, QPushButton {
        font-size: 15px;
    }
    QPushButton {
        padding: 4px 14px;
    }
    """)
    viewer = LogViewer()
    viewer.show()
    sys.exit(app.exec_())

