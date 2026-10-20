"""
Range Pyramid Backtester 
===========================

Instrument-independent range trading research engine.

Input CSV:
    Required: time, open, high, low, close
    Optional: volume, bid, ask

Design:
- The strategy does not contain instrument-specific assumptions.
- Symbol/instrument properties are supplied through Config.
- Supports any FX pair (and can be adapted to other instruments).
- Uses a rolling support/resistance range.
- Opens long positions near the lower range zone.
- Opens short positions near the upper range zone.
- Initial and pyramid positions use the configured micro lot size.
- Adds only to a profitable basket and only after favorable movement.
- Scales out most of the basket as price approaches the opposite range side.
- Closes the remainder near the range boundary.
- Invalidates the range on decisive breakouts.
- Applies spread and commission assumptions when supplied.
- Enforces daily loss and maximum drawdown protections.
- Produces trade-level and equity-level CSV reports.

IMPORTANT:
This is a research/backtesting engine. It does not connect to a broker
or place real orders.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Instrument identity
    symbol: str = "EURUSD"
    base_currency: str = "EUR"
    quote_currency: str = "USD"

    # Price / contract model
    # For standard FX:
    #   contract_size = 100000
    #   lot_size = 0.01 means 1,000 base units
    contract_size: float = 100000.0
    lot_size: float = 0.01
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 100.0

    # Account
    starting_balance: float = 100.0
    account_currency: str = "USD"

    # Execution assumptions
    spread_pips: float = 0.0
    commission_per_lot_round_turn: float = 0.0
    slippage_pips: float = 0.0

    # FX pip definition.
    # 0.0001 for most FX pairs; 0.01 for JPY-quoted pairs.
    pip_size: float = 0.0001

    # Range detection
    lookback: int = 48
    atr_period: int = 14
    min_range_atr: float = 2.0
    max_range_atr: float = 12.0

    # Strategy geometry
    # 0.20 = bottom/top 20% of the range.
    entry_zone_pct: float = 0.20

    # Pyramid
    max_stack: int = 5
    add_step_range_pct: float = 0.12
    require_profit_for_add: bool = True

    # Exits
    majority_target_pct: float = 0.80
    majority_close_fraction: float = 0.75
    final_exit_buffer_pct: float = 0.03

    # Protection
    breakout_atr_multiple: float = 0.75
    basket_stop_atr_multiple: float = 1.50

    # Account protection
    max_daily_loss: float | None = None
    max_total_drawdown: float | None = None

    # Data handling
    use_bid_ask_if_available: bool = True


@dataclass
class Position:
    side: str
    entry: float
    lots: float
    entry_time: str


@dataclass
class ClosedTrade:
    entry_time: str
    exit_time: str
    side: str
    entry: float
    exit: float
    lots: float
    gross_pnl: float
    costs: float
    pnl: float
    reason: str


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class RangePyramidBacktester:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.positions: list[Position] = []
        self.closed: list[ClosedTrade] = []

        self.balance = cfg.starting_balance
        self.equity_curve: list[dict] = []

        self.current_range = None
        self.range_id = 0

        self.day_start_balance = cfg.starting_balance
        self.current_day = None
        self.trading_halted_today = False
        self.trading_halted_permanently = False

        self.peak_equity = cfg.starting_balance
        self.max_drawdown = 0.0

    # -----------------------------------------------------------------------
    # Instrument calculations
    # -----------------------------------------------------------------------

    def normalize_lots(self, lots: float) -> float:
        lots = max(self.cfg.min_lot, min(self.cfg.max_lot, lots))

        steps = round(lots / self.cfg.lot_step)
        normalized = steps * self.cfg.lot_step

        return round(
            max(self.cfg.min_lot, min(self.cfg.max_lot, normalized)), 8
        )

    def pip_value_per_lot(self) -> float:
        """
        Approximate quote-currency pip value for FX:

            contract_size * pip_size

        For EURUSD:
            100000 * 0.0001 = $10/pip/lot

        For USDJPY:
            the raw quote-currency pip value is not USD; a production
            broker-connected implementation should use MT5 tick_value.
        """
        return self.cfg.contract_size * self.cfg.pip_size

    def price_move_value(self, price_difference: float, lots: float) -> float:
        return price_difference * lots * self.cfg.contract_size

    def execution_price(self, side: str, price: float, is_entry: bool) -> float:
        """
        Apply a simple spread/slippage model.

        If the input is a mid/close price:
          long entry  -> ask
          short entry -> bid
          long exit   -> bid
          short exit  -> ask
        """
        half_spread = (self.cfg.spread_pips * self.cfg.pip_size) / 2
        slip = self.cfg.slippage_pips * self.cfg.pip_size

        if side == "long":
            adjustment = half_spread + (slip if is_entry else -slip)
            return price + adjustment

        adjustment = half_spread + (slip if is_entry else -slip)
        return price - adjustment

    def trade_cost(self, lots: float) -> float:
        return lots * self.cfg.commission_per_lot_round_turn

    # -----------------------------------------------------------------------
    # Indicators / range
    # -----------------------------------------------------------------------

    @staticmethod
    def atr(df: pd.DataFrame, period: int) -> pd.Series:
        prev_close = df["close"].shift(1)

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.rolling(period).mean()

    def detect_range(self, df: pd.DataFrame, i: int):
        if i < max(self.cfg.lookback, self.cfg.atr_period):
            return None

        window = df.iloc[i - self.cfg.lookback:i]

        support = float(window["low"].min())
        resistance = float(window["high"].max())
        width = resistance - support

        atr = float(df.iloc[i]["atr"])

        if not np.isfinite(atr) or atr <= 0 or width <= 0:
            return None

        width_atr = width / atr

        if not (
            self.cfg.min_range_atr
            <= width_atr
            <= self.cfg.max_range_atr
        ):
            return None

        self.range_id += 1

        return {
            "support": support,
            "resistance": resistance,
            "width": width,
            "atr": atr,
            "id": self.range_id,
        }

    # -----------------------------------------------------------------------
    # Basket calculations
    # -----------------------------------------------------------------------

    def basket_side(self):
        if not self.positions:
            return None
        return self.positions[0].side

    def basket_lots(self):
        return sum(p.lots for p in self.positions)

    def basket_avg_entry(self):
        if not self.positions:
            return None

        total_lots = self.basket_lots()

        return sum(
            p.entry * p.lots for p in self.positions
        ) / total_lots

    def basket_pnl(self, price: float) -> float:
        pnl = 0.0

        for p in self.positions:
            direction = 1 if p.side == "long" else -1
            pnl += self.price_move_value(
                (price - p.entry) * direction,
                p.lots,
            )

        return pnl

    # -----------------------------------------------------------------------
    # Position operations
    # -----------------------------------------------------------------------

    def open_position(self, side: str, price: float, timestamp):
        if len(self.positions) >= self.cfg.max_stack:
            return False

        lots = self.normalize_lots(self.cfg.lot_size)

        execution = self.execution_price(
            side,
            price,
            is_entry=True,
        )

        self.positions.append(
            Position(
                side=side,
                entry=float(execution),
                lots=lots,
                entry_time=str(timestamp),
            )
        )

        return True

    def close_fraction(
        self,
        price: float,
        timestamp,
        fraction: float,
        reason: str,
    ):
        if not self.positions:
            return

        fraction = max(0.0, min(1.0, fraction))

        remaining = []
        to_close = []

        for p in self.positions:
            close_lots = p.lots * fraction
            remain_lots = p.lots - close_lots

            close_lots = self.normalize_lots(close_lots) \
                if close_lots >= self.cfg.min_lot else 0.0

            if close_lots > p.lots:
                close_lots = p.lots

            if close_lots > 1e-10:
                execution = self.execution_price(
                    p.side,
                    price,
                    is_entry=False,
                )

                direction = 1 if p.side == "long" else -1

                gross_pnl = self.price_move_value(
                    (execution - p.entry) * direction,
                    close_lots,
                )

                costs = self.trade_cost(close_lots)
                pnl = gross_pnl - costs

                self.balance += pnl

                to_close.append(
                    ClosedTrade(
                        entry_time=p.entry_time,
                        exit_time=str(timestamp),
                        side=p.side,
                        entry=p.entry,
                        exit=execution,
                        lots=close_lots,
                        gross_pnl=gross_pnl,
                        costs=costs,
                        pnl=pnl,
                        reason=reason,
                    )
                )

            actual_remaining = p.lots - close_lots

            if actual_remaining > 1e-10:
                remaining.append(
                    Position(
                        p.side,
                        p.entry,
                        actual_remaining,
                        p.entry_time,
                    )
                )

        self.positions = remaining
        self.closed.extend(to_close)

    # -----------------------------------------------------------------------
    # Strategy rules
    # -----------------------------------------------------------------------

    def should_add(self, price: float, rng: dict) -> bool:
        if not self.positions:
            return False

        if len(self.positions) >= self.cfg.max_stack:
            return False

        side = self.basket_side()
        avg = self.basket_avg_entry()

        if self.cfg.require_profit_for_add:
            if side == "long" and price <= avg:
                return False

            if side == "short" and price >= avg:
                return False

        step = rng["width"] * self.cfg.add_step_range_pct
        last_entry = self.positions[-1].entry

        if side == "long":
            return price >= last_entry + step

        return price <= last_entry - step

    def daily_loss_hit(self) -> bool:
        if self.cfg.max_daily_loss is None:
            return False

        daily_pnl = self.balance - self.day_start_balance

        return daily_pnl <= -abs(self.cfg.max_daily_loss)

    def total_drawdown_hit(self, equity: float) -> bool:
        if self.cfg.max_total_drawdown is None:
            return False

        return (
            self.peak_equity - equity
        ) >= abs(self.cfg.max_total_drawdown)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self, df: pd.DataFrame):
        df = df.copy()

        required = {
            "time",
            "open",
            "high",
            "low",
            "close",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce",
        )

        if df["time"].isna().all():
            raise ValueError(
                "The 'time' column could not be parsed."
            )

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df = (
            df.dropna(
                subset=["time", "open", "high", "low", "close"]
            )
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        df["atr"] = self.atr(
            df,
            self.cfg.atr_period,
        )

        for i in range(len(df)):
            row = df.iloc[i]

            timestamp = row["time"]
            price = float(row["close"])

            # Daily reset.
            day = timestamp.date()

            if self.current_day != day:
                self.current_day = day
                self.day_start_balance = self.balance
                self.trading_halted_today = False

            atr = (
                float(row["atr"])
                if np.isfinite(row["atr"])
                else np.nan
            )

            rng = self.detect_range(df, i)

            if rng is not None:
                self.current_range = rng

            rng = self.current_range

            equity = self.balance

            if self.positions:
                equity += self.basket_pnl(price)

            self.peak_equity = max(
                self.peak_equity,
                equity,
            )

            drawdown = self.peak_equity - equity
            self.max_drawdown = max(
                self.max_drawdown,
                drawdown,
            )

            self.equity_curve.append(
                {
                    "time": timestamp,
                    "balance": self.balance,
                    "equity": equity,
                    "drawdown": drawdown,
                    "open_positions": len(self.positions),
                    "range_support": (
                        rng["support"] if rng else np.nan
                    ),
                    "range_resistance": (
                        rng["resistance"] if rng else np.nan
                    ),
                }
            )

            # Account-level kill switch.
            if self.total_drawdown_hit(equity):
                self.trading_halted_permanently = True

                if self.positions:
                    self.close_fraction(
                        price,
                        timestamp,
                        1.0,
                        "max_total_drawdown",
                    )

                continue

            # Daily loss protection.
            if self.daily_loss_hit():
                self.trading_halted_today = True

                if self.positions:
                    self.close_fraction(
                        price,
                        timestamp,
                        1.0,
                        "max_daily_loss",
                    )

                continue

            if (
                rng is None
                or not np.isfinite(atr)
                or self.trading_halted_today
                or self.trading_halted_permanently
            ):
                continue

            support = rng["support"]
            resistance = rng["resistance"]
            width = rng["width"]

            # Bottom/top zone defined by one configurable percentage.
            lower_entry = (
                support
                + width * self.cfg.entry_zone_pct
            )

            upper_entry = (
                resistance
                - width * self.cfg.entry_zone_pct
            )

            # 80% target by default.
            long_target = (
                support
                + width * self.cfg.majority_target_pct
            )

            short_target = (
                resistance
                - width * self.cfg.majority_target_pct
            )

            # Breakout invalidation.
            long_break = (
                price
                > resistance
                + atr * self.cfg.breakout_atr_multiple
            )

            short_break = (
                price
                < support
                - atr * self.cfg.breakout_atr_multiple
            )

            # ---------------------------------------------------------------
            # Manage existing basket
            # ---------------------------------------------------------------

            if self.positions:
                side = self.basket_side()
                avg = self.basket_avg_entry()

                if side == "long":
                    stop = (
                        avg
                        - atr * self.cfg.basket_stop_atr_multiple
                    )

                    if price <= stop or short_break:
                        self.close_fraction(
                            price,
                            timestamp,
                            1.0,
                            "range_break_or_stop",
                        )
                        continue

                    if price >= long_target:
                        self.close_fraction(
                            price,
                            timestamp,
                            self.cfg.majority_close_fraction,
                            "majority_target",
                        )

                    if self.positions and (
                        price
                        >= resistance
                        - width * self.cfg.final_exit_buffer_pct
                    ):
                        self.close_fraction(
                            price,
                            timestamp,
                            1.0,
                            "range_resistance",
                        )

                elif side == "short":
                    stop = (
                        avg
                        + atr * self.cfg.basket_stop_atr_multiple
                    )

                    if price >= stop or long_break:
                        self.close_fraction(
                            price,
                            timestamp,
                            1.0,
                            "range_break_or_stop",
                        )
                        continue

                    if price <= short_target:
                        self.close_fraction(
                            price,
                            timestamp,
                            self.cfg.majority_close_fraction,
                            "majority_target",
                        )

                    if self.positions and (
                        price
                        <= support
                        + width * self.cfg.final_exit_buffer_pct
                    ):
                        self.close_fraction(
                            price,
                            timestamp,
                            1.0,
                            "range_support",
                        )

            # ---------------------------------------------------------------
            # New entry
            # ---------------------------------------------------------------

            if not self.positions:
                if price <= lower_entry and not short_break:
                    self.open_position(
                        "long",
                        price,
                        timestamp,
                    )

                elif price >= upper_entry and not long_break:
                    self.open_position(
                        "short",
                        price,
                        timestamp,
                    )

            # ---------------------------------------------------------------
            # Pyramid
            # ---------------------------------------------------------------

            elif self.should_add(price, rng):
                self.open_position(
                    self.basket_side(),
                    price,
                    timestamp,
                )

        # Close anything remaining at end of test.
        if self.positions:
            final_row = df.iloc[-1]

            self.close_fraction(
                float(final_row["close"]),
                final_row["time"],
                1.0,
                "end_of_test",
            )

        equity_df = pd.DataFrame(
            self.equity_curve
        )

        trades_df = pd.DataFrame(
            [asdict(x) for x in self.closed]
        )

        # ---------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------

        total_pnl = (
            self.balance
            - self.cfg.starting_balance
        )

        if not trades_df.empty:
            wins = trades_df[
                trades_df["pnl"] > 0
            ]

            losses = trades_df[
                trades_df["pnl"] < 0
            ]

            gross_profit = wins["pnl"].sum()
            gross_loss = abs(losses["pnl"].sum())

            trade_count = len(trades_df)

            win_rate = (
                len(wins) / trade_count * 100
                if trade_count
                else 0.0
            )

            profit_factor = (
                gross_profit / gross_loss
                if gross_loss
                else math.inf
            )

            avg_trade = (
                trades_df["pnl"].mean()
            )

            largest_win = (
                trades_df["pnl"].max()
            )

            largest_loss = (
                trades_df["pnl"].min()
            )

            total_lots = (
                trades_df["lots"].sum()
            )

        else:
            gross_profit = 0.0
            gross_loss = 0.0
            trade_count = 0
            win_rate = 0.0
            profit_factor = math.inf
            avg_trade = 0.0
            largest_win = 0.0
            largest_loss = 0.0
            total_lots = 0.0

        metrics = {
            "symbol": self.cfg.symbol,
            "starting_balance": self.cfg.starting_balance,
            "ending_balance": self.balance,
            "net_pnl": total_pnl,
            "return_pct": (
                total_pnl
                / self.cfg.starting_balance
                * 100
            ),
            "closed_trade_parts": trade_count,
            "winning_trade_parts": (
                len(wins)
                if not trades_df.empty
                else 0
            ),
            "losing_trade_parts": (
                len(losses)
                if not trades_df.empty
                else 0
            ),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "average_trade": avg_trade,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "total_lots_traded": total_lots,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": (
                self.max_drawdown
                / self.peak_equity
                * 100
                if self.peak_equity
                else 0.0
            ),
            "max_open_positions": (
                int(
                    equity_df["open_positions"].max()
                )
                if not equity_df.empty
                else 0
            ),
        }

        return (
            equity_df,
            trades_df,
            pd.Series(metrics),
        )

# ---------------------------------------------------------------------------
# Parquet loader
# ---------------------------------------------------------------------------

def load_parquet(path: str) -> pd.DataFrame:
    """
    Load a processed market-bar Parquet dataset.

    The processed bar schema preserves:

        - Bid OHLC
        - Ask OHLC
        - Mid OHLC
        - Spread statistics
        - Tick count

    The backtester uses the mid-price OHLC as the primary
    analytical price series while retaining bid/ask and
    spread information for execution-aware analysis.
    """

    df = pd.read_parquet(path)

    # --------------------------------------------------------
    # Required canonical columns
    # --------------------------------------------------------

    required = {
        "symbol",
        "timestamp",

        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",

        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",

        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",

        "spread_mean",
        "spread_min",
        "spread_max",

        "tick_count",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Processed Parquet dataset is missing "
            f"required columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Normalize timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # Strategy-facing OHLC
    #
    # Use mid-price as the primary market price.
    # Preserve bid/ask/spread columns alongside it.
    # --------------------------------------------------------

    df["time"] = df["timestamp"]

    df["open"] = df["mid_open"]
    df["high"] = df["mid_high"]
    df["low"] = df["mid_low"]
    df["close"] = df["mid_close"]

    # --------------------------------------------------------
    # Volume compatibility
    #
    # Tick data does not provide traded volume.
    # tick_count is therefore the appropriate proxy.
    # --------------------------------------------------------

    df["volume"] = df["tick_count"]

    # --------------------------------------------------------
    # Final ordering
    #
    # Keep the strategy-facing columns first, followed by
    # the full market-data information.
    # --------------------------------------------------------

    columns = [
        "symbol",
        "time",

        "open",
        "high",
        "low",
        "close",

        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",

        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",

        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",

        "spread_mean",
        "spread_min",
        "spread_max",

        "tick_count",
        "volume",
    ]

    df = df[columns]

    return df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Instrument-independent range "
            "pyramiding backtester v2"
        )
    )

    parser.add_argument(
        "csv",
        help="CSV containing OHLC data",
    )

    parser.add_argument(
        "--out",
        default="range_bot_results",
        help="Output directory",
    )

    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Instrument symbol",
    )

    parser.add_argument(
        "--starting-balance",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--lot-size",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--entry-zone",
        type=float,
        default=0.20,
        help="Range entry zone as decimal, e.g. 0.20",
    )

    parser.add_argument(
        "--max-stack",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--spread-pips",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--commission-per-lot",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--slippage-pips",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--pip-size",
        type=float,
        default=0.0001,
        help="0.0001 for most FX pairs, 0.01 for JPY quotes",
    )

    args = parser.parse_args()

    cfg = Config(
        symbol=args.symbol,
        starting_balance=args.starting_balance,
        lot_size=args.lot_size,
        lookback=args.lookback,
        entry_zone_pct=args.entry_zone,
        max_stack=args.max_stack,
        spread_pips=args.spread_pips,
        commission_per_lot_round_turn=args.commission_per_lot,
        slippage_pips=args.slippage_pips,
        pip_size=args.pip_size,
    )

    df = load_parquet(args.csv)

    bot = RangePyramidBacktester(cfg)

    equity, trades, metrics = bot.run(df)

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        out / "trades.csv",
        index=False,
    )

    equity.to_csv(
        out / "equity.csv",
        index=False,
    )

    metrics.to_csv(
        out / "metrics.csv",
        header=["value"],
    )

    print(
        "\n=== RANGE PYRAMIDING BACKTEST v2 ==="
    )

    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isinf(value):
                print(
                    f"{key:24s}: inf"
                )
            else:
                print(
                    f"{key:24s}: {value:.6f}"
                )
        else:
            print(
                f"{key:24s}: {value}"
            )

    print(
        f"\nResults written to: {out.resolve()}"
    )

# ---------------------------------------------------------------------------
# Pipeline interface
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str,
    timeframe: str,
    run_id: int | str | None = None,
    starting_balance: float = 100.0,
    lot_size: float = 0.01,
    lookback: int = 48,
    entry_zone: float = 0.20,
    max_stack: int = 5,
    spread_pips: float = 0.0,
    commission_per_lot: float = 0.0,
    slippage_pips: float = 0.0,
    pip_size: float = 0.0001,
) -> dict:
    """
    Run the range-pyramid backtester from the main application pipeline.

    The backtester consumes OHLC bars generated by the data pipeline.

    Expected location:

        data/processed/<symbol>/<timeframe>/

    The function deliberately contains orchestration only.
    Strategy logic remains inside RangePyramidBacktester.
    """

    project_root = Path(__file__).resolve().parents[2]

    processed_dir = (
        project_root
        / "data"
        / "processed"
        / symbol
        / timeframe
    )

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory does not exist:\n"
            f"{processed_dir}"
        )

    # -----------------------------------------------------------------------
    # Locate generated bar dataset
    # -----------------------------------------------------------------------

    parquet_files = sorted(
        processed_dir.glob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No processed OHLC Parquet files found in:\n"
            f"{processed_dir}"
        )

    if len(parquet_files) > 1:
        print(
            f"[BACKTEST] Multiple bar files found. "
            f"Using: {parquet_files[0].name}"
        )

    input_file = parquet_files[0]

    print()
    print("=== BACKTEST INPUT ===")
    print(f"Symbol:     {symbol}")
    print(f"Timeframe:  {timeframe}")
    print(f"Input:      {input_file}")

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    cfg = Config(
        symbol=symbol,
        starting_balance=starting_balance,
        lot_size=lot_size,
        lookback=lookback,
        entry_zone_pct=entry_zone,
        max_stack=max_stack,
        spread_pips=spread_pips,
        commission_per_lot_round_turn=commission_per_lot,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
    )

    # -----------------------------------------------------------------------
    # Load bars
    # -----------------------------------------------------------------------

    df = load_parquet(
        str(input_file)
    )

    # -----------------------------------------------------------------------
    # Run engine
    # -----------------------------------------------------------------------

    bot = RangePyramidBacktester(cfg)

    equity, trades, metrics = bot.run(df)

    # -----------------------------------------------------------------------
    # Results directory
    # -----------------------------------------------------------------------

    results_dir = (
        project_root
        / "data"
        / "results"
        / "range_pyramid"
        / (
            f"run-{run_id}"
            if run_id is not None
            else "latest"
        )
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------------

    trades.to_csv(
        results_dir / "trades.csv",
        index=False,
    )

    equity.to_csv(
        results_dir / "equity.csv",
        index=False,
    )

    metrics.to_csv(
        results_dir / "metrics.csv",
        header=["value"],
    )

    print()
    print("=== BACKTEST COMPLETE ===")
    print(f"Input:       {input_file}")
    print(f"Results:     {results_dir}")
    print(f"Net PnL:     {metrics['net_pnl']:.6f}")
    print(f"Return:      {metrics['return_pct']:.4f}%")
    print(f"Trades:      {metrics['closed_trade_parts']}")
    print(f"Win rate:    {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor: {metrics['profit_factor']}")
    print(f"Max drawdown:{metrics['max_drawdown']:.6f}")

    return {
    "results_path": str(results_dir),
    "input_path": str(input_file),
    "metrics": metrics,
}

if __name__ == "__main__":
    main()
