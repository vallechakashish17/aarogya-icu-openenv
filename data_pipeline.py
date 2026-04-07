import pandas as pd
import time

class MedicalDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.current_index = 0

    def get_live_feed(self):
        """Simulates a live admission feed from the hospital."""
        if self.current_index < len(self.df):
            patient = self.df.iloc[self.current_index].to_dict()
            self.current_index += 1
            return patient
        else:
            # If we reach the end, loop back to simulate a busy hospital
            self.current_index = 0
            return self.df.iloc[self.current_index].to_dict()

    def get_emergency_status(self, patient_id):
        """Checks if a specific patient is currently in a critical state."""
        # In a real app, this would query a live database
        pass