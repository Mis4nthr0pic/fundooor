import sqlite3
import logging
import argparse
from web3 import Web3
from web3.middleware import geth_poa_middleware
from tqdm import tqdm

# Constants
RPC_ENDPOINT = "https://api.testnet.abs.xyz"
CHAIN_ID = 11124
CONTRACT_ADDRESS = Web3.to_checksum_address("0x6790724c1188ca7141ef57a9ad861b686292a147")
MINT_FUNCTION_ABI = {
    "inputs": [
        {"internalType": "address", "name": "to", "type": "address"},
        {"internalType": "uint256", "name": "id", "type": "uint256"},
        {"internalType": "uint256", "name": "amount", "type": "uint256"},
        {"internalType": "bytes", "name": "data", "type": "bytes"}
    ],
    "name": "mint",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
}
DB_PATH = "distribution.db"

class NFTMinter:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.contract = self.web3.eth.contract(address=CONTRACT_ADDRESS, abi=[MINT_FUNCTION_ABI])
        self.setup_logging()
        self.setup_database()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler("minting.log"), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Initialize the mint table in the database."""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS mint (
                        address TEXT PRIMARY KEY,
                        status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'minted', 'failed')),
                        tx_hash TEXT,
                        token_id INTEGER,
                        amount INTEGER,
                        minted_at DATETIME,
                        FOREIGN KEY(address) REFERENCES wallets(address)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS mint_control (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        status TEXT DEFAULT 'running' CHECK(status IN ('running', 'stopped')),
                        last_address TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(last_address) REFERENCES wallets(address)
                    )
                ''')
                
                # Initialize control status if not exists
                conn.execute('''
                    INSERT OR IGNORE INTO mint_control (id, status)
                    VALUES (1, 'running')
                ''')
                
                # Initialize mint table with receiving wallets if empty
                conn.execute('''
                    INSERT OR IGNORE INTO mint (address)
                    SELECT address FROM wallets 
                    WHERE wallet_type = 'receiving'
                    AND address NOT IN (SELECT address FROM mint)
                ''')
                
                conn.commit()
            except sqlite3.Error as e:
                self.logger.error(f"Database error during setup: {e}")
                raise

    def mint_nfts(self, token_id: int, amount: int = 1, data: bytes = b'test'):
        """Mint NFTs for all pending wallets."""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Convert 'test' to bytes32
                data_bytes32 = data.ljust(32, b'\0')

                # Get control status
                control = conn.execute('SELECT status, last_address FROM mint_control WHERE id = 1').fetchone()
                if control and control[0] == 'stopped' and control[1]:
                    # Resume from last address
                    self.logger.info(f"Resuming from address: {control[1]}")
                    where_clause = f"AND w.address >= '{control[1]}'"
                else:
                    where_clause = ""

                # Get all pending receiving wallets with their private keys
                receiving_wallets = conn.execute(f'''
                    SELECT w.address, w.private_key
                    FROM wallets w
                    JOIN mint m ON w.address = m.address
                    WHERE w.wallet_type = 'receiving'
                    AND m.status = 'pending'
                    {where_clause}
                    ORDER BY w.address
                ''').fetchall()

                if not receiving_wallets:
                    self.logger.info("No pending wallets found for minting")
                    return

                # Set control status to running
                conn.execute('''
                    UPDATE mint_control 
                    SET status = 'running',
                        last_address = NULL,
                        updated_at = datetime('now')
                    WHERE id = 1
                ''')
                conn.commit()

                # Initialize progress bar
                progress_bar = tqdm(receiving_wallets, desc="Minting NFTs")

                try:
                    for address, private_key in progress_bar:
                        progress_bar.set_description(f"Minting to {address}")
                        
                        # Check if minting was stopped
                        control = conn.execute('SELECT status FROM mint_control WHERE id = 1').fetchone()
                        if control and control[0] == 'stopped':
                            self.logger.info("Minting stopped by user")
                            return

                        # Mint NFT to self
                        tx_hash = self.mint(address, token_id, amount, data_bytes32, private_key)
                        if tx_hash:
                            self.logger.info(f"Minted NFT for {address} with tx hash: {tx_hash}")
                            # Update mint table to mark as minted
                            conn.execute('''
                                UPDATE mint
                                SET status = 'minted',
                                    tx_hash = ?,
                                    token_id = ?,
                                    amount = ?,
                                    minted_at = datetime('now')
                                WHERE address = ?
                            ''', (tx_hash, token_id, amount, address))
                        else:
                            self.logger.error(f"Failed to mint NFT for {address}")
                            conn.execute('''
                                UPDATE mint
                                SET status = 'failed',
                                    minted_at = datetime('now')
                                WHERE address = ?
                            ''', (address,))

                        # Update last processed address
                        conn.execute('''
                            UPDATE mint_control
                            SET last_address = ?,
                                updated_at = datetime('now')
                            WHERE id = 1
                        ''', (address,))
                        conn.commit()

                except KeyboardInterrupt:
                    self.logger.info("Minting interrupted by user")
                    # Update control status to stopped
                    conn.execute('''
                        UPDATE mint_control
                        SET status = 'stopped',
                            updated_at = datetime('now')
                        WHERE id = 1
                    ''')
                    conn.commit()
                    return

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during minting: {e}")
                raise

    def mint(self, to_address: str, token_id: int, amount: int, data: bytes, private_key: str) -> str:
        """Execute the mint transaction."""
        try:
            account = self.web3.eth.account.from_key(private_key)
            nonce = self.web3.eth.get_transaction_count(account.address)
            
            # Convert to checksum address
            to_address = self.web3.to_checksum_address(to_address)

            # Get current gas price from network
            gas_price = self.web3.eth.gas_price
            
            # Build initial transaction for gas estimation
            transaction = self.contract.functions.mint(to_address, token_id, amount, data).build_transaction({
                'chainId': CHAIN_ID,
                'from': account.address,
                'nonce': nonce,
                'gasPrice': gas_price,
                'value': 0
            })

            # Estimate gas for this specific transaction
            estimated_gas = self.web3.eth.estimate_gas(transaction)
            
            # Add 10% buffer to estimated gas
            gas_limit = int(estimated_gas * 1.1)

            # Update transaction with estimated gas
            transaction.update({
                'gas': gas_limit
            })

            # Calculate and log the total gas cost
            total_gas_cost = gas_price * gas_limit
            self.logger.info(f"Estimated gas: {estimated_gas}")
            self.logger.info(f"Gas limit with buffer: {gas_limit}")
            self.logger.info(f"Gas price: {self.web3.from_wei(gas_price, 'gwei')} Gwei")
            self.logger.info(f"Total gas cost: {self.web3.from_wei(total_gas_cost, 'ether')} ETH")

            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            return self.web3.to_hex(tx_hash)

        except Exception as e:
            self.logger.error(f"Error minting NFT: {e}")
            return None

    def show_minting_status(self):
        """Display the current minting status for all wallets."""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                status = conn.execute('''
                    SELECT 
                        status,
                        COUNT(*) as count
                    FROM mint
                    GROUP BY status
                ''').fetchall()

                print("\nMinting Status Summary:")
                for status_type, count in status:
                    print(f"{status_type}: {count}")

                # Show last 5 minted addresses
                last_minted = conn.execute('''
                    SELECT address, tx_hash, minted_at
                    FROM mint
                    WHERE status = 'minted'
                    ORDER BY minted_at DESC
                    LIMIT 5
                ''').fetchall()

                if last_minted:
                    print("\nLast 5 Minted Addresses:")
                    for addr, tx_hash, minted_at in last_minted:
                        print(f"Address: {addr}")
                        print(f"Tx Hash: {tx_hash}")
                        print(f"Minted At: {minted_at}")
                        print("-" * 50)

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise

    def restart_minting(self, token_id: int = 1, amount: int = 1):
        """Reset minting status for all wallets to pending and start minting."""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Reset all minting statuses
                conn.execute('''
                    UPDATE mint
                    SET status = 'pending',
                        tx_hash = NULL,
                        minted_at = NULL
                ''')
                
                # Reset control status
                conn.execute('''
                    UPDATE mint_control
                    SET status = 'running',
                        last_address = NULL,
                        updated_at = datetime('now')
                    WHERE id = 1
                ''')
                
                conn.commit()
                self.logger.info("Reset all wallets to pending status")
                self.show_minting_status()
                
                # Start minting process
                self.logger.info(f"Starting minting process for all wallets (token_id: {token_id}, amount: {amount})")
                self.mint_nfts(token_id=token_id, amount=amount)
                
            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise

    def stop_minting(self):
        """Stop the minting process."""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                conn.execute('''
                    UPDATE mint_control
                    SET status = 'stopped',
                        updated_at = datetime('now')
                    WHERE id = 1
                ''')
                conn.commit()
                self.logger.info("Minting process stopped")
                self.show_minting_status()
            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise

    def resume_minting(self, token_id: int, amount: int = 1):
        """Resume minting from the last processed address."""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                control = conn.execute('SELECT status, last_address FROM mint_control WHERE id = 1').fetchone()
                if control and control[1]:
                    self.logger.info(f"Resuming minting from address: {control[1]}")
                    self.mint_nfts(token_id=token_id, amount=amount)
                else:
                    self.logger.info("No resume point found, starting from beginning")
                    self.mint_nfts(token_id=token_id, amount=amount)
            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise

def main():
    parser = argparse.ArgumentParser(description='NFT Minting System')
    parser.add_argument('--mint', action='store_true', help='Start minting NFTs')
    parser.add_argument('--token-id', type=int, default=1, help='Token ID to mint')
    parser.add_argument('--amount', type=int, default=1, help='Amount of tokens to mint')
    parser.add_argument('--status', action='store_true', help='Show minting status')
    parser.add_argument('--restart', action='store_true', help='Restart minting for all wallets')
    parser.add_argument('--stop', action='store_true', help='Stop the minting process')
    parser.add_argument('--resume', action='store_true', help='Resume minting from last position')
    
    args = parser.parse_args()
    minter = NFTMinter()
    
    if args.mint:
        minter.mint_nfts(token_id=args.token_id, amount=args.amount)
    elif args.status:
        minter.show_minting_status()
    elif args.restart:
        minter.restart_minting(token_id=args.token_id, amount=args.amount)
    elif args.stop:
        minter.stop_minting()
    elif args.resume:
        minter.resume_minting(token_id=args.token_id, amount=args.amount)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 