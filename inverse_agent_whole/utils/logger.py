import json
import datetime
import os
from typing import Any, Dict

class RunLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.events = []
        self.start_time = datetime.datetime.now()
        
        # Directory Structure
        self.dirs = {
            "root": output_dir,
            "artifacts": os.path.join(output_dir, "artifacts"),
            "traces": os.path.join(output_dir, "traces"),
            "logs": os.path.join(output_dir, "logs")
        }
        
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        self.main_log_path = os.path.join(self.dirs["logs"], "pipeline_events.jsonl")

    def log(self, event_type: str, content: Any, meta: Dict = None):
        """Log a structured event to the main timeline."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": event_type,
            "content": str(content) if not isinstance(content, (dict, list, int, float, bool, str)) else content,
            "meta": meta or {}
        }
        self.events.append(entry)
        
        # Append to JSONL file immediately
        try:
            with open(self.main_log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[RunLogger] Error writing event: {e}")

    def log_artifact(self, name: str, content: Any, ext: str = "json"):
        """Save a specific artifact (plan, code, result) to a file."""
        filename = f"{name}.{ext}"
        path = os.path.join(self.dirs["artifacts"], filename)
        
        try:
            with open(path, 'w') as f:
                if ext == "json" and isinstance(content, (dict, list)):
                    json.dump(content, f, indent=2)
                else:
                    f.write(str(content))
        except Exception as e:
            print(f"[RunLogger] Error saving artifact {name}: {e}")
            
    def log_trace(self, phase: str, role: str, content: str):
        """Log detailed LLM traces (Thinking, Prompt, Response)."""
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"{phase}_{timestamp}_{role}.md"
        path = os.path.join(self.dirs["traces"], filename)
        
        try:
            with open(path, 'w') as f:
                f.write(f"# {phase} - {role}\n\n")
                f.write(str(content))
        except Exception as e:
            print(f"[RunLogger] Error saving trace {filename}: {e}")
