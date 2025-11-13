"""Interactive CSV plotter for the 6roomsOffice dataset.

Usage:
  python plotter.py --file path/to/dataset.csv

If no file is provided the script will try to find a `.csv` in the same
directory as this script and use the first match.

The app creates a small Dash web app with:
- X-axis selector (defaults to `timestamp` if available)
- Y-axis selector (one or more columns)
- Date range selector (limits the range by the `timestamp` column)

Dependencies:
  pandas, dash, plotly

Run the app then open http://127.0.0.1:8050 in your browser.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import pandas as pd
import plotly.express as px

from dash import Dash, dcc, html, Input, Output


def find_csv_file(provided: Optional[str]) -> Optional[str]:
	if provided:
		if os.path.exists(provided):
			return provided
		raise FileNotFoundError(f"CSV file not found: {provided}")

	# try to find a csv in current script dir
	script_dir = os.path.dirname(__file__)
	candidates = glob.glob(os.path.join(script_dir, "*.csv"))
	if candidates:
		return candidates[0]

	# try parent folder
	parent = os.path.abspath(os.path.join(script_dir, os.pardir))
	candidates = glob.glob(os.path.join(parent, "*.csv"))
	if candidates:
		return candidates[0]

	return None


def load_csv(path: str) -> pd.DataFrame:
	# Try parsing timestamp column if present
	df = pd.read_csv(path)
	# normalize column names by stripping whitespace
	df.columns = [c.strip() for c in df.columns]

	if "timestamp" in df.columns:
		try:
			df["timestamp"] = pd.to_datetime(df["timestamp"])
			df = df.sort_values("timestamp").reset_index(drop=True)
		except Exception:
			# leave as-is if parsing fails
			pass

	return df


def make_dash_app(df: pd.DataFrame, csv_path: str) -> Dash:
	# determine columns
	cols = list(df.columns)
	default_x = "timestamp" if "timestamp" in cols else cols[0]

	app = Dash(__name__)

	# date range defaults (only if timestamp exists and is datetime)
	min_date = None
	max_date = None
	if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
		min_date = df["timestamp"].min().date()
		max_date = df["timestamp"].max().date()

	app.layout = html.Div(
		[
			html.H3(f"Interactive plotter — {os.path.basename(csv_path)}"),
			html.Div([
				html.Label("X axis:"),
				dcc.Dropdown(id="xcol", options=[{"label": c, "value": c} for c in cols], value=default_x, clearable=False),
			], style={"width": "48%", "display": "inline-block"}),

			html.Div([
				html.Label("Y axis (choose one or multiple):"),
				dcc.Dropdown(id="ycols", options=[{"label": c, "value": c} for c in cols if c != default_x], multi=True, value=[cols[1]] if len(cols) > 1 else [cols[0]]),
			], style={"width": "48%", "display": "inline-block", "float": "right"}),

			html.Div([
				html.Label("Date window (applies when `timestamp` is selected):"),
				dcc.DatePickerRange(id="daterange", min_date_allowed=min_date, max_date_allowed=max_date, start_date=min_date, end_date=max_date),
			], style={"marginTop": "10px"}),

			html.Div([
				html.Label("Downsample points (for performance):"),
				dcc.Slider(id="downsample", min=1, max=1000, step=1, value=1,
						   marks={1: "1", 10: "10", 100: "100", 500: "500", 1000: "1000"}),
			], style={"marginTop": "12px"}),

			dcc.Graph(id="main-graph", style={"height": "70vh"}),

			html.Div(id="status", style={"marginTop": "6px", "fontSize": "12px", "color": "#666"}),
		], style={"width": "95%", "margin": "auto"},
	)


	@app.callback(
		Output("main-graph", "figure"),
		Output("status", "children"),
		Input("xcol", "value"),
		Input("ycols", "value"),
		Input("daterange", "start_date"),
		Input("daterange", "end_date"),
		Input("downsample", "value"),
	)
	def update(xcol, ycols, start_date, end_date, downsample):
		if not ycols:
			return px.line(), "No Y columns selected"

		# ensure ycols is a list
		if isinstance(ycols, str):
			ycols = [ycols]

		dff = df.copy()
		if xcol == "timestamp" and "timestamp" in dff.columns and pd.api.types.is_datetime64_any_dtype(dff["timestamp"]):
			if start_date:
				dff = dff[dff["timestamp"] >= pd.to_datetime(start_date)]
			if end_date:
				# include entire end day
				dff = dff[dff["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
			x = dff["timestamp"]
		else:
			x = dff[xcol]

		# downsample by taking every nth row
		n = max(1, int(downsample))
		if n > 1:
			dff = dff.iloc[::n]
			x = x.iloc[::n] if hasattr(x, "iloc") else x

		fig = px.line()
		for col in ycols:
			if col not in dff.columns:
				continue
			try:
				fig.add_scatter(x=x, y=dff[col], mode="lines", name=col)
			except Exception:
				# fallback: try converting to numeric
				y = pd.to_numeric(dff[col], errors="coerce")
				fig.add_scatter(x=x, y=y, mode="lines", name=col)

		fig.update_layout(margin={"l": 40, "r": 20, "t": 40, "b": 40})

		status = f"Showing {len(dff)} rows | x={xcol} | y={', '.join(ycols)}"
		return fig, status

	return app


def main():
	parser = argparse.ArgumentParser(description="Interactive plotter for AAU 6roomsOffice CSV")
	parser.add_argument("--file", "-f", help="Path to CSV file")
	parser.add_argument("--host", default="127.0.0.1", help="Dash host")
	parser.add_argument("--port", type=int, default=8050, help="Dash port")
	args = parser.parse_args()

	csv_path = find_csv_file(args.file)
	if not csv_path:
		print("No CSV file found. Provide one with --file path/to/file.csv")
		return

	print(f"Loading CSV: {csv_path}")
	df = load_csv(csv_path)

	app = make_dash_app(df, csv_path)
	print(f"Starting Dash app at http://{args.host}:{args.port}")
	app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
	main()

