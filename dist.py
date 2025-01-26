import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import csv
import sqlite3
import logging
import random
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from eth_account import Account
from web3 import Web3, exceptions
from web3.types import TxParams
from web3.middleware import geth_poa_middleware
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Network Configuration
RPC_ENDPOINT = "http://localhost:8545"
CHAIN_ID = 31337

# File Paths
DB_PATH = "distribution.db"
DB_LOG_PATH = "distribution.log"
FUNDING_CSV = "funding.csv"
RECEIVING_CSV = "receiving.csv"

# Distribution Parameters
MIN_SENDS_PER_WALLET = 25
MAX_SENDS_PER_WALLET = 35
DEFAULT_ETH_AMOUNT = 0.00002
MIN_FUNDING_BALANCE = 0.00002
MAX_TX_RETRIES = 3
GAS_LIMIT = 21000

# Anvil specific gas price (much lower than mainnet)
DEFAULT_GAS_PRICE = 1000000000  # 1 gwei

# New constants for mainnet mode and delays
MAINNET_MODE = False  # Set to True for mainnet
TX_DELAY = 1  # Delay between transactions in seconds for testnet
MAINNET_TX_DELAY = 3  # Delay between transactions in seconds for mainnet
BATCH_SIZE = 50  # Number of transactions before a longer pause
BATCH_PAUSE = 30  # Pause duration in seconds after each batch

class Distributor:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.setup_database()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(DB_LOG_PATH), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS distribution_plan (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    funding_wallet TEXT NOT NULL,
                    total_receivers INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS distribution_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER,
                    funding_wallet TEXT NOT NULL,
                    receiving_wallet TEXT NOT NULL,
                    amount_eth TEXT NOT NULL,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'sent', 'failed')),
                    tx_hash TEXT,
                    executed_at DATETIME,
                    FOREIGN KEY(plan_id) REFERENCES distribution_plan(id)
                );

                CREATE TABLE IF NOT EXISTS wallets (
                    address TEXT PRIMARY KEY,
                    wallet_type TEXT CHECK(wallet_type IN ('funding', 'receiving')),
                    private_key TEXT NOT NULL,
                    current_balance TEXT DEFAULT '0',
                    last_updated DATETIME
                );
            ''')

    def import_wallets(self):
        """
        Import funding and receiving wallets from CSV files into the database.
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Import funding wallets
                with open(FUNDING_CSV, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) != 2:
                            self.logger.error(f"Invalid row in funding.csv: {row}")
                            continue

                        address, private_key = row
                        address = str(address).strip()
                        private_key = str(private_key).strip()

                        if not self.web3.is_address(address):
                            self.logger.error(f"Invalid Ethereum address: {address}")
                            continue

                        balance_wei = self.web3.eth.get_balance(address)
                        balance_eth = self.web3.from_wei(balance_wei, "ether")
                        
                        conn.execute('''
                            INSERT OR REPLACE INTO wallets 
                            (address, wallet_type, private_key, current_balance, last_updated)
                            VALUES (?, 'funding', ?, ?, datetime('now'))
                        ''', (address, private_key, str(balance_eth)))  # Store as string
                        self.logger.info(f"Imported funding wallet: {address}")

                # Import receiving wallets and check their balances
                with open(RECEIVING_CSV, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) != 2:
                            self.logger.error(f"Invalid row in receiving.csv: {row}")
                            continue

                        address, private_key = row
                        address = str(address).strip()
                        private_key = str(private_key).strip()

                        if not self.web3.is_address(address):
                            self.logger.error(f"Invalid Ethereum address: {address}")
                            continue

                        balance_wei = self.web3.eth.get_balance(address)
                        balance_eth = self.web3.from_wei(balance_wei, "ether")

                        conn.execute('''
                            INSERT OR REPLACE INTO wallets 
                            (address, wallet_type, private_key, current_balance, last_updated)
                            VALUES (?, 'receiving', ?, ?, datetime('now'))
                        ''', (address, private_key, str(balance_eth)))  # Store as string
                        self.logger.info(f"Imported receiving wallet: {address}")

                conn.commit()
                self.logger.info("Wallets imported successfully.")
                
            except FileNotFoundError as e:
                self.logger.error(f"CSV file not found: {e}")
                raise
            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during wallet import: {e}")
                raise


    def show_distribution_status(self):
        """
        Display the current status of distribution plans and last processed wallet.
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get last processed wallet
                last_processed = conn.execute('''
                    SELECT id, funding_wallet, total_receivers, status, created_at
                    FROM distribution_plan 
                    ORDER BY id DESC LIMIT 1
                ''').fetchone()

                if last_processed:
                    print("\nLast Processed Plan:")
                    headers = ['Plan ID', 'Funding Wallet', 'Total Receivers', 'Status', 'Created At']
                    print(tabulate([[
                        last_processed[0],
                        last_processed[1],
                        last_processed[2],
                        last_processed[3],
                        last_processed[4]
                    ]], headers=headers, tablefmt='grid'))

                # Get distribution summary
                summary = conn.execute('''
                    SELECT 
                        status,
                        COUNT(*) as count,
                        SUM(total_receivers) as total_receivers
                    FROM distribution_plan 
                    GROUP BY status
                ''').fetchall()

                if summary:
                    print("\nDistribution Plans Summary:")
                    headers = ['Status', 'Number of Plans', 'Total Receivers']
                    print(tabulate(summary, headers=headers, tablefmt='grid'))
                else:
                    print("\nNo distribution plans found.")

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error showing status: {e}")
                raise

    def create_distribution_plan(self, eth_amount: float = DEFAULT_ETH_AMOUNT):
        """
        Create a distribution plan that assigns receiving wallets to funding wallets.
        
        Args:
            eth_amount: Amount of ETH to send to each wallet (default: DEFAULT_ETH_AMOUNT)
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Calculate total cost per transaction (transfer amount + max gas cost)
                gas_cost_wei = DEFAULT_GAS_PRICE * GAS_LIMIT
                transfer_amount_wei = self.web3.to_wei(eth_amount, 'ether')  # Use passed eth_amount
                total_cost_per_tx_wei = gas_cost_wei + transfer_amount_wei
                total_cost_per_tx_eth = float(self.web3.from_wei(total_cost_per_tx_wei, 'ether'))

                # Get all receiving wallets
                receiving_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'receiving'
                ''').fetchall()
                receiving_addresses = [w[0] for w in receiving_wallets]
                total_receivers = len(receiving_addresses)

                if not receiving_addresses:
                    self.logger.error("No receiving wallets found")
                    return

                # Get funding wallets with sufficient balance for at least MIN_SENDS_PER_WALLET transactions
                min_required_balance = float(total_cost_per_tx_eth * MIN_SENDS_PER_WALLET)
                
                # Debug log to check the value
                self.logger.info(f"Minimum required balance: {min_required_balance} ETH")
                
                funding_wallets = conn.execute('''
                    SELECT address, current_balance 
                    FROM wallets 
                    WHERE wallet_type = 'funding' 
                    AND CAST(current_balance AS FLOAT) >= ?
                    ORDER BY address
                ''', (min_required_balance,)).fetchall()

                if not funding_wallets:
                    self.logger.error(f"No funding wallets with sufficient balance found. Each wallet needs at least {min_required_balance} ETH")
                    return

                # Calculate max transactions possible for each wallet
                wallet_capacities = []
                for _, balance_str in funding_wallets:
                    balance = float(balance_str)  # Convert string balance to float
                    max_txs = min(
                        int(balance / total_cost_per_tx_eth),  # Max txs possible with available balance
                        MAX_SENDS_PER_WALLET  # Cap at maximum allowed sends
                    )
                    wallet_capacities.append(max_txs)

                # Calculate optimal distribution
                remaining_receivers = total_receivers
                receivers_per_wallet = []
                
                for i in range(len(funding_wallets)):
                    if i == len(funding_wallets) - 1:
                        # Last wallet gets remaining receivers if it has capacity
                        receivers = min(remaining_receivers, wallet_capacities[i])
                    else:
                        # Calculate optimal number of receivers for this wallet
                        avg_remaining = remaining_receivers // (len(funding_wallets) - i)
                        receivers = min(
                            max(MIN_SENDS_PER_WALLET, avg_remaining),
                            MAX_SENDS_PER_WALLET,
                            wallet_capacities[i]  # Cannot exceed wallet's capacity
                        )
                    
                    receivers_per_wallet.append(receivers)
                    remaining_receivers -= receivers

                if remaining_receivers > 0:
                    self.logger.error(f"Not enough funding wallet capacity to handle all receivers. {remaining_receivers} receivers left unassigned")
                    return

                # Create distribution plans
                cursor = conn.cursor()
                current_receiver_index = 0

                for (funding_wallet, balance), num_receivers in zip(funding_wallets, receivers_per_wallet):
                    # Insert plan into database
                    cursor.execute('''
                        INSERT INTO distribution_plan 
                        (funding_wallet, total_receivers, status)
                        VALUES (?, ?, 'pending')
                    ''', (funding_wallet, num_receivers))
                    plan_id = cursor.lastrowid

                    # Select receiving wallets for this funding wallet
                    end_index = current_receiver_index + num_receivers
                    plan_receivers = receiving_addresses[current_receiver_index:end_index]
                    current_receiver_index = end_index

                    # Create distribution tasks
                    for receiver in plan_receivers:
                        cursor.execute('''
                            INSERT INTO distribution_tasks 
                            (plan_id, funding_wallet, receiving_wallet, amount_eth, status)
                            VALUES (?, ?, ?, ?, 'pending')
                        ''', (plan_id, funding_wallet, receiver, str(eth_amount)))  # Store as string

                    total_cost = num_receivers * total_cost_per_tx_eth
                    self.logger.info(f"Created distribution plan {plan_id} for {funding_wallet} "
                                   f"to send to {num_receivers} receivers. Total cost: {total_cost:.6f} ETH")
                    conn.commit()

                self.logger.info(f"Distribution plan created successfully. "
                               f"Total receivers assigned: {current_receiver_index}")

                # Print updated status
                print("\nUpdated status after creating new plans:")
                self.show_distribution_status()

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error creating distribution plan: {e}")
                raise

    def send_transaction(self, private_key: str, to_address: str, amount_eth: float) -> Optional[str]:
        """
        Send ETH transaction and return transaction hash if successful.
        
        Args:
            private_key: Sender's private key
            to_address: Recipient's address
            amount_eth: Amount of ETH to send
            
        Returns:
            Optional[str]: Transaction hash if successful, None if failed
        """
        try:
            account = Account.from_key(private_key)
            from_address = account.address
            nonce = self.web3.eth.get_transaction_count(from_address)
            
            # Use fixed gas price for Anvil
            transaction: TxParams = {
                'nonce': nonce,
                'to': to_address,
                'value': self.web3.to_wei(amount_eth, 'ether'),
                'gas': GAS_LIMIT,
                'gasPrice': DEFAULT_GAS_PRICE,  # Use fixed low gas price
                'chainId': CHAIN_ID
            }

            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            return self.web3.to_hex(tx_hash)

        except Exception as e:
            self.logger.error(f"Error sending transaction: {str(e)}")
            return None

    def execute_distribution(self):
        """
        Execute pending distribution tasks and update wallet balances.
        Implements delays and batching for mainnet safety.
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                pending_plans = conn.execute('''
                    SELECT id, funding_wallet 
                    FROM distribution_plan 
                    WHERE status = 'pending'
                    ORDER BY id
                ''').fetchall()

                if not pending_plans:
                    self.logger.info("No pending distribution plans found")
                    return

                tx_delay = MAINNET_TX_DELAY if MAINNET_MODE else TX_DELAY
                tx_count = 0

                for plan_id, funding_wallet in pending_plans:
                    conn.execute('''
                        UPDATE distribution_plan 
                        SET status = 'in_progress' 
                        WHERE id = ?
                    ''', (plan_id,))
                    conn.commit()

                    private_key = conn.execute('''
                        SELECT private_key 
                        FROM wallets 
                        WHERE address = ?
                    ''', (funding_wallet,)).fetchone()[0]

                    tasks = conn.execute('''
                        SELECT id, receiving_wallet, amount_eth 
                        FROM distribution_tasks 
                        WHERE plan_id = ? AND status = 'pending'
                        ORDER BY id
                    ''', (plan_id,)).fetchall()

                    success_count = 0
                    fail_count = 0

                    for task_id, receiving_wallet, amount_eth in tqdm(tasks, desc=f"Processing plan {plan_id}"):
                        # Check if we need a batch pause
                        if MAINNET_MODE and tx_count > 0 and tx_count % BATCH_SIZE == 0:
                            self.logger.info(f"Batch of {BATCH_SIZE} complete. Pausing for {BATCH_PAUSE} seconds...")
                            time.sleep(BATCH_PAUSE)

                        retry_count = 0
                        tx_hash = None

                        while retry_count < MAX_TX_RETRIES and not tx_hash:
                            if retry_count > 0:
                                time.sleep(tx_delay * 2)  # Double delay on retries
                                self.logger.info(f"Retry {retry_count} for task {task_id}")

                            tx_hash = self.send_transaction(
                                private_key=private_key,
                                to_address=receiving_wallet,
                                amount_eth=float(amount_eth)  # Convert string to float
                            )
                            retry_count += 1

                        if tx_hash:
                            # Update task status and balances
                            conn.execute('''
                                UPDATE distribution_tasks 
                                SET status = 'sent', 
                                    tx_hash = ?,
                                    executed_at = datetime('now') 
                                WHERE id = ?
                            ''', (tx_hash, task_id))

                            # Update wallet balances
                            funding_balance_wei = self.web3.eth.get_balance(funding_wallet)
                            receiving_balance_wei = self.web3.eth.get_balance(receiving_wallet)
                            
                            funding_balance_eth = str(self.web3.from_wei(funding_balance_wei, "ether"))
                            receiving_balance_eth = str(self.web3.from_wei(receiving_balance_wei, "ether"))

                            conn.execute('''
                                UPDATE wallets 
                                SET current_balance = ?,
                                    last_updated = datetime('now')
                                WHERE address = ?
                            ''', (funding_balance_eth, funding_wallet))

                            conn.execute('''
                                UPDATE wallets 
                                SET current_balance = ?,
                                    last_updated = datetime('now')
                                WHERE address = ?
                            ''', (receiving_balance_eth, receiving_wallet))

                            success_count += 1
                            tx_count += 1

                            # Add delay between successful transactions
                            if tx_count < len(tasks):  # Don't delay after last transaction
                                time.sleep(tx_delay)
                        else:
                            conn.execute('''
                                UPDATE distribution_tasks 
                                SET status = 'failed',
                                executed_at = datetime('now') 
                                WHERE id = ?
                            ''', (task_id,))
                            fail_count += 1

                        conn.commit()

                    final_status = 'completed' if fail_count == 0 else 'failed'
                    conn.execute('''
                        UPDATE distribution_plan 
                        SET status = ? 
                        WHERE id = ?
                    ''', (final_status, plan_id))
                    conn.commit()

                    self.logger.info(f"Plan {plan_id} completed: {success_count} successful, {fail_count} failed")

                self.show_distribution_status()

            except Exception as e:
                self.logger.error(f"Error executing distribution: {str(e)}")
                raise

    def check_wallet_balance(self, address: str) -> float:
        """
        Check ETH balance for a given wallet address.
        
        Args:
            address: Ethereum wallet address
            
        Returns:
            float: Balance in ETH
            
        Raises:
            ValueError: If address is invalid
        """
        try:
            if not self.web3.is_address(address):
                raise ValueError(f"Invalid Ethereum address: {address}")
            
            balance_wei = self.web3.eth.get_balance(address)
            balance_eth = float(self.web3.from_wei(balance_wei, "ether"))
            
            print(f"\nWallet Balance:")
            print(f"Address: {address}")
            print(f"Balance: {balance_eth:.6f} ETH")
            
            return balance_eth
            
        except Exception as e:
            self.logger.error(f"Error checking balance for {address}: {str(e)}")
            raise

    def check_all_receiving_balances(self):
        """
        Check ETH balance for all receiving wallets and display in a table format.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Get all receiving wallets
                receiving_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'receiving'
                    ORDER BY address
                ''').fetchall()

                if not receiving_wallets:
                    self.logger.error("No receiving wallets found")
                    return

                # Get balances
                balances = []
                total_balance = 0
                for (address,) in tqdm(receiving_wallets, desc="Checking balances"):
                    balance_wei = self.web3.eth.get_balance(address)
                    balance_eth = float(self.web3.from_wei(balance_wei, "ether"))
                    balances.append([address, f"{balance_eth:.6f}"])
                    total_balance += balance_eth

                # Display results
                print("\nReceiving Wallet Balances:")
                headers = ['Address', 'Balance (ETH)']
                print(tabulate(balances, headers=headers, tablefmt='grid'))
                
                # Display summary
                print(f"\nSummary:")
                print(f"Total wallets: {len(receiving_wallets)}")
                print(f"Total balance: {total_balance:.6f} ETH")
                print(f"Average balance: {(total_balance/len(receiving_wallets)):.6f} ETH")

        except Exception as e:
            self.logger.error(f"Error checking receiving wallet balances: {str(e)}")
            raise

    def update_distribution_amount(self, new_amount: float):
        """
        Update the ETH amount for all pending distribution tasks.
        
        Args:
            new_amount: New amount of ETH to send for each pending transaction
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Check if there are any pending tasks
                pending_count = conn.execute('''
                    SELECT COUNT(*) 
                    FROM distribution_tasks 
                    WHERE status = 'pending'
                ''').fetchone()[0]

                if pending_count == 0:
                    self.logger.info("No pending tasks found to update")
                    return

                # Update all pending tasks with new amount
                conn.execute('''
                    UPDATE distribution_tasks 
                    SET amount_eth = ?
                    WHERE status = 'pending'
                ''', (str(new_amount),))  # Store as string
                
                conn.commit()
                
                self.logger.info(f"Updated {pending_count} pending tasks to new amount: {new_amount} ETH")

                # Show updated distribution status
                self.show_distribution_status()

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error updating distribution amount: {str(e)}")
                raise

    def resend_to_all(self, new_amount: float):
        """
        Reset all tasks to pending and update their amount, then execute distribution.
        
        Args:
            new_amount: New amount of ETH to send for each transaction
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Reset all tasks to pending and update amount
                conn.execute('''
                    UPDATE distribution_tasks 
                    SET status = 'pending',
                        amount_eth = ?,
                        tx_hash = NULL,
                        executed_at = NULL
                ''', (str(new_amount),))

                # Reset all plans to pending
                conn.execute('''
                    UPDATE distribution_plan 
                    SET status = 'pending'
                ''')
                
                conn.commit()
                
                total_tasks = conn.execute('SELECT COUNT(*) FROM distribution_tasks').fetchone()[0]
                self.logger.info(f"Reset {total_tasks} tasks to pending with new amount: {new_amount} ETH")

                # Show updated status
                self.show_distribution_status()

                # Execute the distribution
                print("\nStarting distribution with new amount...")
                self.execute_distribution()

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error resending to all wallets: {str(e)}")
                raise

def main():
    parser = argparse.ArgumentParser(description='ETH Distribution System')
    parser.add_argument('--import-wallets', action='store_true', help='Import wallets from CSV files')
    parser.add_argument('--create-plan', action='store_true', help='Create distribution plan')
    parser.add_argument('--execute', action='store_true', help='Execute distribution')
    parser.add_argument('--status', action='store_true', help='Show distribution status')
    parser.add_argument('--check-balance', type=str, help='Check ETH balance for a wallet address')
    parser.add_argument('--check-receiving', action='store_true', help='Check balance of all receiving wallets')
    parser.add_argument('--amount', type=float, help='Amount of ETH to send to each wallet (default: 0.00002)', default=DEFAULT_ETH_AMOUNT)
    parser.add_argument('--update-amount', type=float, help='Update amount for pending distribution tasks')
    parser.add_argument('--resend-all', type=float, help='Reset all tasks and resend with new amount')
    
    args = parser.parse_args()
    distributor = Distributor()
    
    if args.import_wallets:
        distributor.import_wallets()
    elif args.create_plan:
        distributor.create_distribution_plan(eth_amount=args.amount)
    elif args.execute:
        distributor.execute_distribution()
    elif args.status:
        distributor.show_distribution_status()
    elif args.check_balance:
        distributor.check_wallet_balance(args.check_balance)
    elif args.check_receiving:
        distributor.check_all_receiving_balances()
    elif args.update_amount:
        distributor.update_distribution_amount(args.update_amount)
    elif args.resend_all:
        distributor.resend_to_all(args.resend_all)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()