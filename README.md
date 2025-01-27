# ETH Distribution Bot

A Python-based system for managing and automating ETH distributions from funding wallets to receiving wallets. Supports both local testing (Anvil) and mainnet operations with configurable batch processing and safety delays.

test minter: 0x6790724c1188ca7141ef57a9ad861b686292a147

## Requirements & Installation

1. Install Python dependencies:
```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

2. Prepare wallet files:
- `funding.csv`: Contains funding wallet addresses and private keys
- `receiving.csv`: Contains receiving wallet addresses and private keys

CSV format:
```csv
address,private_key
0x123...,0xabc...
```

## Usage

### Basic Commands

1. Import wallets from CSV files:
```bash
python3 dist.py --import-wallets
```

2. Create a distribution plan:
```bash
# With default amount (0.00002 ETH)
python3 dist.py --create-plan

# With custom amount
python3 dist.py --create-plan --amount 0.00004
```

3. Execute the distribution:
```bash
python3 dist.py --execute
```

### Monitoring & Management

Check distribution status:
```bash
python3 dist.py --status
```

Check wallet balances:
```bash
# Check single wallet
python3 dist.py --check-balance 0x123...

# Check all receiving wallets
python3 dist.py --check-receiving
```

### Update Distribution Amounts

Update pending transactions:
```bash
python3 dist.py --update-amount 0.00004
```

Reset and resend all transactions:
```bash
python3 dist.py --resend-all 0.00004
```

## Configuration

Key parameters in `dist.py`:

```python
# Network Configuration
RPC_ENDPOINT = "http://localhost:8545"  # Update for mainnet
CHAIN_ID = 31337  # 1 for mainnet

# Distribution Parameters
MIN_SENDS_PER_WALLET = 25
MAX_SENDS_PER_WALLET = 35
DEFAULT_ETH_AMOUNT = 0.00002
MIN_FUNDING_BALANCE = 0.00002
GAS_LIMIT = 21000

# Mainnet Safety Parameters
MAINNET_MODE = False  # Set to True for mainnet
TX_DELAY = 1  # Seconds between transactions (testnet)
MAINNET_TX_DELAY = 3  # Seconds between transactions (mainnet)
BATCH_SIZE = 50  # Transactions per batch
BATCH_PAUSE = 30  # Seconds between batches
```

## Mainnet Usage

For mainnet operations:

1. Update network settings in `dist.py`:
```python
RPC_ENDPOINT = "your-mainnet-rpc"
CHAIN_ID = 1
MAINNET_MODE = True
```

2. Consider adjusting safety parameters:
- Increase `MAINNET_TX_DELAY` for more time between transactions
- Adjust `BATCH_SIZE` and `BATCH_PAUSE` based on network conditions
- Review gas settings for current network conditions

## Safety Features

- Configurable delays between transactions
- Batch processing with automatic pauses
- Pre-distribution balance checks
- Transaction retry logic
- Comprehensive error handling
- Database transaction safety
- Real-time balance updates

## Database & Logging

- SQLite database (`distribution.db`) tracks:
  - Distribution plans
  - Individual transactions
  - Wallet balances and status

- Logging (`distribution.log`) includes:
  - Wallet imports
  - Plan creation
  - Transaction execution
  - Errors and warnings

## Error Handling

The system handles:
- Invalid wallet addresses
- Insufficient balances
- Failed transactions
- Network issues
- Database errors

## File Structure

- `dist.py`: Main distribution script
- `wallet_gen.py`: Wallet generation utility
- `funding.csv`: Funding wallet details
- `receiving.csv`: Receiving wallet details
- `distribution.db`: SQLite database
- `distribution.log`: Operation logs
- `requirements.txt`: Python dependencies

## Support

For issues or questions:
1. Check the logs in `distribution.log`
2. Verify wallet balances and network status
3. Ensure CSV files are properly formatted
4. Check database integrity

## Resetting the Database and Retesting

To start fresh and retest the distribution process, you can reset the database by following these steps:

1. **Delete the Database File**:
   - Remove the `distribution.db` file to clear all existing data.
   - You can do this manually or run the following command in your terminal:
     ```bash
     rm distribution.db
     ```

2. **Re-import Wallets**:
   - Use the following command to import wallets from the CSV files:
     ```bash
     python3 dist.py --import-wallets
     ```

3. **Create New Distribution Plans**:
   - Create new distribution plans with the default or custom ETH amount:
     ```bash
     python3 dist.py --create-plan
     # Or with a custom amount
     python3 dist.py --create-plan --amount 0.00004
     ```

4. **Execute Distribution**:
   - Run the distribution process:
     ```bash
     python3 dist.py --execute
     ```

This process will ensure that you are working with a clean slate and can accurately test the distribution functionality.

## NFT Minting

The system includes NFT minting functionality through the `minter.py` script. This allows you to mint NFTs to all receiving wallets with full control over the minting process.

### Minting Commands

1. Start minting NFTs:
```bash
# With default token ID (1) and amount (1)
python3 minter.py --mint

# With custom token ID and amount
python3 minter.py --mint --token-id 2 --amount 1
```

2. Control minting process:
```bash
# Check minting status
python3 minter.py --status

# Stop minting (can be resumed later)
python3 minter.py --stop

# Resume minting from last position
python3 minter.py --resume

# Restart minting for all wallets
python3 minter.py --restart
```

### Minting Features

- **Stop/Resume**: You can stop the minting process at any time (Ctrl+C or --stop) and resume later
- **Progress Tracking**: Shows real-time progress and maintains a log of all operations
- **Status Monitoring**: View current minting status and last minted addresses
- **Error Handling**: Failed mints are tracked and can be retried
- **Database Integration**: All minting operations are tracked in the database

### Minting Process

1. The script will mint one NFT to each receiving wallet
2. Each wallet will receive the specified token ID and amount
3. Progress is saved automatically and can be resumed if interrupted
4. Failed mints are tracked and can be retried using --restart
