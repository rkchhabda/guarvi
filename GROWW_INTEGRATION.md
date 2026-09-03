# Groww API Integration for Live Market Data

This project now supports fetching live market data and placing orders via the Groww API.

## Prerequisites

1. **Groww Developer Account**: Sign up at [Groww Developer Portal](https://groww.in/api)
2. **API Credentials**: Generate API Key and Secret from the developer portal
3. **Static IP**: Your server's static IP must be whitelisted in the Groww Developer Portal
4. **Python Package**: `growwapi` (added to requirements.txt)

## Getting a Static IP (for API whitelisting)

Groww only accepts API requests whose **source IP** matches one whitelisted in
the developer portal — it checks the IP your requests arrive from, so your
machine's outbound IP must be fixed. Pick one of these options:

| Option | How | Best for |
|---|---|---|
| **Cloud VPS** (recommended) | AWS EC2 (Elastic IP), GCP, Azure, DigitalOcean, or Oracle Cloud free tier (reserved public IP). Deploy this project there and whitelist the VPS IP. | 24/7 trading bots, full control |
| **ISP static IP** | Ask your broadband provider (Jio/Airtel/ACT/...) for a static IP or business broadband plan. | Running from home/office |
| **VPN with dedicated IP** | Commercial VPNs offering a dedicated/static IP (e.g., NordVPN Dedicated IP, PureVPN, Windscribe Static IP). Connect before running scripts. | Quick setup from anywhere |
| **Static-IP proxy** | Any HTTP(S) proxy with a fixed egress IP. Set `HTTPS_PROXY=http://user:pass@host:port` — both `growwapi` and the IP check honor it. | Existing infrastructure |

> **Note:** Dynamic DNS (DDNS) does **not** help — Groww whitelists the IP
> itself, not a hostname. Residential IPs usually change on every router
> restart, which silently breaks whitelisting.

### Are there free options?

| Option | Verdict |
|---|---|
| **Oracle Cloud Always Free** | ✅ The only genuinely free + reliable route: permanently-free VM (Ampere A1 Arm or AMD micro) with a reserved public IP that survives stop/start. Pick an India region (`ap-mumbai-1` / `ap-hyderabad-1`) for low NSE latency. Caveats: card required at signup (not charged), popular regions often report "out of capacity", idle Always-Free VMs can be reclaimed (a market-hours bot is fine). Verify current terms at signup. |
| AWS Free Tier | ❌ Now credit-based (up to $200 / 6 months) and public IPv4 is billed ~$0.005/hr even when attached to a running instance |
| Google Cloud Free Tier | ❌ e2-micro free tier is US-region-only (~200ms latency to NSE) and external IPv4 is billed hourly even in use |
| Azure Free Tier | ❌ 12-month/credit based; public IP billed |
| Free VPNs (Proton free, etc.) | ❌ Shared, rotating egress IPs — cannot be whitelisted, and Groww may block known VPN ranges |
| "Free VPS" hosts | ❌ Unreliable, short-lived, shared IPs — a risky place to store trading credentials |
| DDNS / Cloudflare Tunnel / Tailscale | ❌ Don't change your outbound IP at all |

### Detect and verify your IP

```bash
# Show current public IP and compare with GROWW_STATIC_IP
python scripts/check_static_ip.py

# Save the detected IP into .env as GROWW_STATIC_IP
python scripts/check_static_ip.py --set

# Machine-readable (for CI/automation)
python scripts/check_static_ip.py --json
```

The loader also auto-warns at `initialize()` if the outbound IP no longer
matches `GROWW_STATIC_IP` (e.g., your ISP changed your IP overnight).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
GROWW_API_KEY=your_actual_api_key
GROWW_SECRET=your_actual_secret
GROWW_STATIC_IP=your_static_ip
```

**Important**: Never commit `.env` to version control!

### 3. Update Configuration

The `config/data.yaml` has been updated with a `groww` section:

```yaml
groww:
  enabled: true
  api_key: ""  # Set via environment variable GROWW_API_KEY
  secret: ""   # Set via environment variable GROWW_SECRET
  static_ip: ""  # Your static IP for API access
  access_token: ""  # Generated from api_key + secret

paths:
  live_dir: data/live/nse
```

## VPS Deployment (Oracle Cloud / any Ubuntu VM)

For fully automated collection of live data during NSE market hours
(09:15–15:30 IST), deploy on a VM whose static IP is whitelisted:

```bash
# On the VM (Ubuntu 22.04/24.04), from the project root:
sudo bash scripts/setup_vps.sh                                  # full setup + systemd
sudo bash scripts/setup_vps.sh --skip-firewall --no-service     # provision only

# Options:
#   --symbols "RELIANCE TCS INFY"    # fetch a custom list (default: --universe nifty100)
#   --candle-interval 5minute        # candle interval for OHLCV
#   --poll-seconds 300               # seconds between fetch cycles
```

The script installs system deps, creates `.venv`, provisions `.env`
(chmod 600), configures `ufw` (SSH rate-limited, outbound open), installs the
`guarvi-live-data` systemd service (enabled, auto-restart), and prints the
VM's public IP to whitelist.

After whitelisting the VM IP in the Groww portal and filling `.env`:

```bash
sudo systemctl start guarvi-live-data
journalctl -u guarvi-live-data -f          # follow logs
sudo systemctl stop guarvi-live-data       # stop
```

**Runner behavior** (`scripts/live_data_runner.py`): polls quotes + candles
every `--poll-seconds` while NSE is open (Mon–Fri, 09:15–15:30 IST), sleeps
outside hours, appends to `data/live/nse/quotes_<date>.csv` and
`data/live/nse/ohlcv/<SYM>_<date>.csv` (deduplicated), and renews the Groww
access token daily. NSE holidays are not special-cased (harmless). Oracle
images ship default iptables REJECT rules for *inbound* traffic — outbound
(what Groww needs) is allowed by default.

## Usage

### Fetch Live Market Data

```bash
# Fetch quotes for specific symbols
python scripts/fetch_live_data.py --symbols RELIANCE TCS INFY

# Fetch OHLCV data for NIFTY100 universe
python scripts/fetch_live_data.py --universe nifty100 --interval 5minute

# Fetch only quotes and save to CSV
python scripts/fetch_live_data.py --symbols RELIANCE --quote-only --save

# Fetch only OHLCV and save to CSV
python scripts/fetch_live_data.py --symbols RELIANCE --ohlcv-only --save
```

### Place Orders

```bash
# Place a market buy order
python scripts/place_order.py --symbol RELIANCE --quantity 1 --side BUY

# Place a limit sell order
python scripts/place_order.py --symbol IDEA --quantity 10 --side SELL --order-type LIMIT --price 15.50

# Dry run (simulate without placing)
python scripts/place_order.py --symbol RELIANCE --quantity 1 --side BUY --dry-run
```

### Programmatic Usage

```python
from src.data.groww_loader import create_groww_loader

# Create loader
loader = create_groww_loader()

# Initialize (generates access token automatically)
if loader.initialize():
    # Get live quote
    quote = loader.get_live_quote("RELIANCE")
    print(f"Last Price: {quote['last_price']}")
    
    # Get live OHLCV
    ohlcv = loader.get_live_ohlcv("RELIANCE", interval="5minute")
    print(ohlcv.tail())
    
    # Place order
    order = loader.place_order(
        trading_symbol="RELIANCE",
        quantity=1,
        transaction_type="BUY",
        order_type="MARKET"
    )
    print(f"Order ID: {order}")
```

## Features

### Live Data
- **Live Quotes**: Real-time price, volume, change, OHLC
- **Live OHLCV**: Intraday candles (1min, 5min, 15min, 30min, 60min, day)
- **Multiple Symbols**: Batch fetch quotes for multiple symbols
- **Data Persistence**: Automatic CSV saving to `data/live/nse/`

### Trading
- **Market Orders**: Instant execution at best available price
- **Limit Orders**: Execute at specified price or better
- **Order Types**: DAY, IOC validity
- **Products**: MIS (intraday), CNC (delivery), NRML (overnight)
- **Exchanges**: NSE, BSE
- **Segments**: CASH, FNO

### Account Info
- **Order Book**: View all orders
- **Positions**: Current open positions
- **Holdings**: Long-term holdings
- **Funds**: Available margin/balance

## Data Flow

```
Groww API (Live) → GrowwLiveLoader → Normalized DataFrames → CSV/Parquet → ML Pipeline
```

The live data can be combined with historical data from `jugaad-data` for:
- Real-time feature engineering
- Live predictions
- Paper trading / backtesting
- Portfolio monitoring

## Rate Limits

Groww API has rate limits. The loader includes:
- Small delays between batch requests (0.1s)
- Retry logic with exponential backoff
- Error handling for rate limit errors

## Troubleshooting

### "Static IP not whitelisted"
- Add your server's public IP to the whitelist in Groww Developer Portal
- Use `curl ifconfig.me` to find your public IP

### "Invalid API key/secret"
- Regenerate credentials in Groww Developer Portal
- Ensure no extra spaces in `.env` file

### "Access token expired"
- Tokens expire daily; the loader regenerates automatically
- Or manually set `GROWW_ACCESS_TOKEN` in `.env`

### "Module not found: growwapi"
```bash
pip install growwapi
```

## Security Best Practices

1. **Never hardcode credentials** - Use environment variables
2. **Rotate API keys periodically** - Generate new keys monthly
3. **Use read-only keys where possible** - For data fetching only
4. **Monitor API usage** - Check Groww Developer Portal for usage stats
5. **Implement circuit breakers** - Stop trading on repeated errors

## Example: Complete Trading Bot

```python
from src.data.groww_loader import create_groww_loader
import time

loader = create_groww_loader()
loader.initialize()

symbol = "RELIANCE"
target_price = 2500.0

while True:
    quote = loader.get_live_quote(symbol)
    if quote and quote['last_price'] <= target_price:
        # Place buy order
        order = loader.place_order(
            trading_symbol=symbol,
            quantity=10,
            transaction_type="BUY",
            order_type="LIMIT",
            price=target_price
        )
        print(f"Buy order placed: {order}")
        break
    
    print(f"Current price: {quote['last_price']}, waiting for {target_price}")
    time.sleep(60)  # Check every minute
```

## Support

- Groww API Documentation: https://groww.in/api/docs
- Issues: Create a GitHub issue in this repository