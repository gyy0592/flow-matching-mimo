import numpy as np

class IterationLogger:
    """A helper class to manage logging and NaN detection within loops."""
    def __init__(self, debug_mode=True, log_first_m=5, log_last_n=5):
        self.history = []
        self.nan_detected = False
        self.debug_mode = debug_mode
        self.log_first_m = log_first_m
        self.log_last_n = log_last_n

    def reset(self, total=None):
        """Clear stored history and NaN flag before a new run."""
        self.history.clear()
        self.nan_detected = False

    def log_and_check_nan(self, **kwargs):
        """
        Logs the metrics for the current step and checks for NaN in the error value.
        If NaN is detected, it prints a debug history.

        Args:
            **kwargs: A dictionary of metrics to log for the current step.

        Returns:
            bool: True if NaN was detected, False otherwise.
        """
        log_step = {key: value for key, value in kwargs.items()}
        self.history.append(log_step)

        error_val = log_step.get("error", float('nan'))

        if np.isnan(error_val) and not self.nan_detected:
            if self.debug_mode:
                print(f"\n\n--- NaN Detected at {log_step.get('header', 'N/A')}! Printing debug history ---")
                self._print_debug_logs()
            self.nan_detected = True
        
        return self.nan_detected

    def _print_debug_logs(self):
        """Prints formatted debug logs based on the stored history."""
        log_buffer = self.history
        if len(log_buffer) <= self.log_first_m + self.log_last_n:
            self._format_and_print(log_buffer)
        else:
            print("--- First M Iterations ---")
            self._format_and_print(log_buffer[:self.log_first_m])
            print("\n...\n")
            print("--- Last N Iterations ---")
            self._format_and_print(log_buffer[-self.log_last_n:])

    def _format_and_print(self, log_buffer):
        """Helper to format and print a buffer of log dictionaries."""
        if not log_buffer:
            return

        headers = [log.get('header', '') for log in log_buffer]
        col_width = 18
        header_line = "Iteration          : " + " -> ".join(f"{h:<{col_width}}" for h in headers)
        print(header_line)
        print("-" * len(header_line))

        def format_log_line(name, key, format_str):
            vals = [format_str.format(log.get(key, float('nan'))) for log in log_buffer]
            return f"{name:<18}: " + " -> ".join(f"{v:<{col_width}}" for v in vals)

        print(format_log_line("Sigma H", "sigma_H", "{:.2f}"))
        print(format_log_line("Sigma X", "sigma_x", "{:.2f}"))
        print(format_log_line("Error", "error", "{:.4e}"))
        print("-" * len(header_line))
        print(format_log_line("H Norm (before)", "h_norm_before", "{:.2f}"))
        print(format_log_line("H Like Score", "h_like_score", "{:.2f}"))
        print(format_log_line("H Vel Term 1", "h_vel_term1", "{:.2e}"))
        print(format_log_line("H Vel Prior Term", "h_vel_prior_term", "{:.2e}"))
        print(format_log_line("H Unconditional Vel", "h_unconditional_vel", "{:.2e}"))
        print(format_log_line("H Correction Vel", "h_correction_vel", "{:.2e}"))
        print(format_log_line("H Vel (pre-clip)", "h_vel_before_clip", "{:.2e}"))
        print(format_log_line("H Vel (post-clip)", "h_vel_after_clip", "{:.2e}"))
        print(format_log_line("H Norm (after)", "h_norm_after", "{:.2f}"))
        print("-" * len(header_line))
        print(format_log_line("X Norm (before)", "x_norm_before", "{:.2f}"))
        print(format_log_line("X Like Score", "x_like_score", "{:.2f}"))
        print(format_log_line("X Vel Term 1", "x_vel_term1", "{:.2e}"))
        print(format_log_line("X Vel Prior Term", "x_vel_prior_term", "{:.2e}"))
        print(format_log_line("X Unconditional Vel", "x_unconditional_vel", "{:.2e}"))
        print(format_log_line("X Correction Vel", "x_correction_vel", "{:.2e}"))
        print(format_log_line("X Velocity", "x_vel", "{:.2e}"))
        print(format_log_line("X Norm (after)", "x_norm_after", "{:.2f}"))
        print("\n")
