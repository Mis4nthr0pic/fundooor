import csv
from datetime import datetime
import os

class TransactionLogger:
    def __init__(self):
        self.log_file = "failed_transactions.csv"
        self.init_log_file()

    def init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'from_address',
                    'to_address',
                    'amount',
                    'nonce',
                    'error_message',
                    'gas_used',
                    'gas_price'
                ])

    def log_failed_tx(self, from_addr, to_addr, amount, nonce, error, gas_used, gas_price):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                from_addr,
                to_addr,
                amount,
                nonce,
                error,
                gas_used,
                gas_price
            ]) 