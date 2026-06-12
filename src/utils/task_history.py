#!/usr/bin/env python3
"""
Task History Manager — 版本化结果存储
======================================
每次训练任务保存到带时间戳的独立子目录，维护统一的历史索引。
展示台 (dashboard) 读取最新任务的结果，同时保留所有历史记录。

用法:
  from src.utils.task_history import TaskRun
  run = TaskRun("survival_v1", base_dir="local_data/Results_adni")
  run.save_csv("model_comparison.csv", df)
  run.save_plot("km_by_risk.png")  # matplotlib figure is already saved
  run.finish({"c_index": 0.745, "auc_3yr": 0.791})
"""

import os, json, shutil
from datetime import datetime

class TaskRun:
    """一次实验运行的结果容器。"""

    def __init__(self, task_name, base_dir="local_data/Results_adni", description=""):
        self.task_name = task_name
        self.base_dir = base_dir
        self.description = description
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.timestamp}_{task_name}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        self.latest_dir = os.path.join(base_dir, f"LATEST_{task_name}")
        self.metrics = {}
        self.start_time = datetime.now()
        os.makedirs(self.run_dir, exist_ok=True)

    def get_path(self, filename):
        """Get full path within this run's directory."""
        return os.path.join(self.run_dir, filename)

    def save_csv(self, filename, df):
        """Save a DataFrame as CSV in this run's directory."""
        path = self.get_path(filename)
        df.to_csv(path, index=False)
        return path

    def save_fig(self, filename):
        """Register a figure that was already saved. Copy it to run dir.
        Call this AFTER plt.savefig() to the original path.
        Pass the path to the saved figure."""
        # If it's a path, copy it
        if os.path.exists(filename):
            dst = os.path.join(self.run_dir, os.path.basename(filename))
            shutil.copy2(filename, dst)
            return dst
        return None

    def save_metrics(self, metrics_dict):
        """Save metrics as JSON for easy retrieval."""
        self.metrics.update(metrics_dict)
        path = os.path.join(self.run_dir, "metrics.json")
        with open(path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        return path

    def finish(self, metrics_dict=None):
        """Mark run as complete: save metrics, update index, and link latest."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        if metrics_dict:
            self.save_metrics(metrics_dict)

        # Update history index
        self._update_index(duration)

        # Update LATEST symlink
        self._link_latest()

        # Print summary
        print(f"\n  📁 Task saved: {self.run_id}")
        print(f"  ⏱  Duration: {duration:.0f}s")
        print(f"  📂 Results:  {self.run_dir}")

    def _update_index(self, duration):
        """Append this run to the shared history index CSV."""
        index_path = os.path.join(self.base_dir, "task_history.csv")

        import pandas as pd
        row = {
            'run_id': self.run_id,
            'task_name': self.task_name,
            'timestamp': self.timestamp,
            'description': self.description,
            'duration_sec': round(duration, 1),
            'run_dir': self.run_dir,
        }
        row.update({f"metric_{k}": v for k, v in self.metrics.items()
                     if isinstance(v, (int, float, str, bool))})

        new_row = pd.DataFrame([row])
        if os.path.exists(index_path):
            existing = pd.read_csv(index_path)
            new_row = pd.concat([existing, new_row], ignore_index=True)
        new_row.to_csv(index_path, index=False)

    def _link_latest(self):
        """Create/update LATEST_<task_name> symlink pointing to this run."""
        if os.path.islink(self.latest_dir) or os.path.exists(self.latest_dir):
            os.remove(self.latest_dir)
        os.symlink(self.run_id, self.latest_dir, target_is_directory=True)

    @staticmethod
    def get_history(base_dir="local_data/Results_adni"):
        """Load the full history index as a DataFrame."""
        import pandas as pd
        index_path = os.path.join(base_dir, "task_history.csv")
        if os.path.exists(index_path):
            return pd.read_csv(index_path)
        return pd.DataFrame()

    @staticmethod
    def get_latest(task_name, base_dir="local_data/Results_adni"):
        """Get path to the latest run of a given task."""
        latest = os.path.join(base_dir, f"LATEST_{task_name}")
        if os.path.islink(latest):
            return os.path.join(base_dir, os.readlink(latest))
        return None

    @staticmethod
    def list_runs(task_name=None, base_dir="local_data/Results_adni"):
        """List all run directories, optionally filtered by task name."""
        runs = []
        for d in sorted(os.listdir(base_dir), reverse=True):
            full = os.path.join(base_dir, d)
            if os.path.isdir(full) and not d.startswith("LATEST_"):
                if task_name is None or d.endswith(task_name):
                    runs.append(full)
        return runs


def print_history(base_dir="local_data/Results_adni"):
    """Print the task history index as a formatted table."""
    import pandas as pd
    idx = pd.read_csv(os.path.join(base_dir, "task_history.csv")) \
        if os.path.exists(os.path.join(base_dir, "task_history.csv")) \
        else pd.DataFrame()

    if idx.empty:
        print("  No task history found.")
        return

    metric_cols = [c for c in idx.columns if c.startswith("metric_")]
    display_cols = ['task_name', 'timestamp', 'description', 'duration_sec'] + metric_cols

    print(f"\n  {'Task':<20} {'Time':<16} {'Duration':<10} {'Key Metrics'}")
    print(f"  {'─'*20} {'─'*16} {'─'*10} {'─'*40}")
    for _, r in idx.iterrows():
        metrics_str = ", ".join(
            f"{c.replace('metric_','')}={r[c]}" for c in metric_cols
            if pd.notna(r[c]) and c in idx.columns
        )
        print(f"  {r['task_name']:<20} {r['timestamp']:<16} "
              f"{r['duration_sec']:.0f}s{'':>4} {metrics_str}")


if __name__ == "__main__":
    print_history()
