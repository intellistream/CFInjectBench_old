import time

class MonthlyTimeBasedStopping:
    def __init__(self, monthly_limits):
        self.start_time = time.time()
        self.monthly_limits = monthly_limits
        self.time_expired = False

    def check_time(self, month):
        max_seconds = self.monthly_limits.get(month)
        elapsed_time = time.time() - self.start_time
        if elapsed_time > max_seconds:
            self.time_expired = True
