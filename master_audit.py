import json
import pandas as pd
from datetime import datetime

class MasterAudit:
    def __init__(self, filename="master_clinical_audit.json"):
        self.filename = filename
        self.audit_id = f"AUDIT-{datetime.now().strftime('%Y%m%d-%H%M')}"
        self.logs = []
        # Mapping actions for the final human-readable report
        self.action_labels = {
            0: "Monitor",
            1: "Administer Medication",
            2: "Provide Oxygen",
            3: "Emergency: Call Doctor"
        }

    def log_step(self, patient_id, step, state, action, reward, info):
        """Records forensic data for every AI decision."""
        entry = {
            "step": step,
            "patient_id": patient_id,
            "vitals": {
                "heart_rate": round(float(state[0]), 2),
                "o2_sat": round(float(state[1]), 2),
                "toxicity": round(float(state[2]), 2)
            },
            "decision": self.action_labels.get(int(action), "Unknown"),
            "clinical_reward": round(reward, 4),
            "emergency_intervention": info.get("doctor_called", False)
        }
        self.logs.append(entry)

    def finalize(self, avg_stability, cmo_feedback):
        """Compiles the final audit and saves to JSON."""
        report = {
            "audit_metadata": {
                "audit_id": self.audit_id,
                "timestamp": datetime.now().isoformat(),
                "overall_stability_score": round(float(avg_stability), 4)
            },
            "cmo_audit_summary": cmo_feedback,
            "forensic_logs": self.logs
        }
        
        with open(self.filename, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"✅ Master Audit File {self.audit_id} finalized and exported to {self.filename}")