# Connecting Google Drive on Kaggle (`T4 x2` / `P100`)

Unlike Google Colab, Kaggle containers do not include a built-in Google Drive mount widget. `env_bootstrap.py` automatically mounts your Drive via `rclone` FUSE whenever a `RCLONE_CONF` secret is present.

## One-Time Setup (Takes ~2 Minutes)

1. **Install `rclone` on your local machine** ([rclone.org/downloads](https://rclone.org/downloads/)) and run:
   ```bash
   rclone config
   ```
2. Create a new remote named **`gdrive`**:
   - Choose **`n`** (New remote)
   - Name: `gdrive`
   - Storage type: **`drive`** (Google Drive)
   - Leave `client_id` and `client_secret` blank (or provide your own Google Cloud OAuth credentials for higher API rate limits)
   - Scope: **`1`** (`drive` — full access)
   - Complete the browser OAuth prompt.
3. **Copy your generated config file**:
   ```bash
   rclone config file
   cat ~/.config/rclone/rclone.conf
   ```
4. **Add the secret to Kaggle**:
   - Open your Kaggle Notebook -> **Add-ons** -> **Secrets** -> **Add Secret**.
   - **Label:** `RCLONE_CONF`
   - **Value:** Paste the full contents of `rclone.conf` (`[gdrive]` section + token JSON).
   - Toggle the secret **ON** for the notebook and enable **Internet** in the Kaggle Session Options sidebar.