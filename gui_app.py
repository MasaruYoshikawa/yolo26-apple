#!/usr/bin/env python3
"""
りんご検出デスクトップアプリケーション (Apple Detection Desktop Application)
YOLOを使用した画像データセット内のりんご自動検出・カウント用GUI（日本語版）
"""

import os
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.scrolledtext as scrolledtext

# detect.pyから検出用コアロジックをインポート
try:
    from detect import run_detection
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from detect import run_detection


class AppleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("りんご物体検出・個数カウントシステム")
        self.root.geometry("900x700")
        self.root.minsize(800, 650)
        
        # 実行パスの解決 (PyInstallerでパッケージ化された場合にも対応)
        if getattr(sys, 'frozen', False):
            self.workspace_dir = Path(sys.executable).parent.resolve()
        else:
            self.workspace_dir = Path(__file__).parent.resolve()
            
        default_dataset = self.workspace_dir / "apple_dataset" / "images" / "test"
        if not default_dataset.exists():
            default_dataset = self.workspace_dir
            
        default_model = self.workspace_dir / "weights" / "best.pt"
        if not default_model.exists():
            default_model = self.workspace_dir / "yolo11n.pt"

        # 状態変数
        self.source_dir_var = tk.StringVar(value=str(default_dataset))
        self.model_path_var = tk.StringVar(value=str(default_model))
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.device_var = tk.StringVar(value="cpu")
        self.imgsz_var = tk.StringVar(value="640")
        self.output_dir_var = tk.StringVar(value=str(self.workspace_dir / "runs" / "detect" / "predictions"))
        
        # スレッド制御変数
        self.processing_thread = None
        self.is_running = False

        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        """UIのスタイルとカラーパレットを設定 (Meiryoフォントを使用してモダンなダークテーマを構成)"""
        self.bg_color = "#1e1e2e"       # ダークスレートパープル
        self.card_bg = "#252538"        # カード背景色
        self.fg_color = "#cdd6f4"       # テキストカラー（オフホワイト）
        self.accent_color = "#89b4fa"   # アクセント（ソフトブルー）
        self.success_color = "#a6e3a1"  # 成功（ソフトグリーン）
        self.warning_color = "#f9e2af"  # 警告（ソフトイエロー）
        self.text_dark = "#11111b"      # ボタンテキスト用ダークカラー
        self.border_color = "#45475a"   # ボーダーカラー
        
        self.root.configure(bg=self.bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # フォント設定
        font_name = "Meiryo UI"
        
        # 各種コンポーネントのスタイル定義
        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, borderwidth=1, relief="solid")
        
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=(font_name, 10))
        style.configure("Card.TLabel", background=self.card_bg, foreground=self.fg_color, font=(font_name, 10))
        style.configure("Header.TLabel", background=self.bg_color, foreground=self.accent_color, font=(font_name, 16, "bold"))
        style.configure("CardHeader.TLabel", background=self.card_bg, foreground=self.accent_color, font=(font_name, 12, "bold"))
        style.configure("StatsVal.TLabel", background=self.card_bg, foreground=self.success_color, font=("Consolas", 14, "bold"))
        style.configure("StatsLbl.TLabel", background=self.card_bg, foreground=self.fg_color, font=(font_name, 9))

        # ボタンのスタイル定義
        style.configure("TButton", font=(font_name, 10, "bold"), borderwidth=0, focuscolor="none")
        style.map("TButton",
                  background=[("active", self.border_color), ("!disabled", self.card_bg)],
                  foreground=[("active", self.fg_color), ("!disabled", self.fg_color)])

        style.configure("Action.TButton", font=(font_name, 11, "bold"), background=self.success_color, foreground=self.text_dark)
        style.map("Action.TButton",
                  background=[("active", "#b4f4af"), ("!disabled", self.success_color)],
                  foreground=[("active", self.text_dark), ("!disabled", self.text_dark)])

        style.configure("Open.TButton", font=(font_name, 10, "bold"), background=self.accent_color, foreground=self.text_dark)
        style.map("Open.TButton",
                  background=[("active", "#99c4ff"), ("!disabled", self.accent_color)],
                  foreground=[("active", self.text_dark), ("!disabled", self.text_dark)])

        # 入力フィールドとコンボボックスのスタイル定義
        style.configure("TEntry", fieldbackground=self.card_bg, foreground=self.fg_color, bordercolor=self.border_color, darkcolor=self.border_color, lightcolor=self.border_color)
        style.configure("TCombobox", fieldbackground=self.card_bg, foreground=self.fg_color, arrowcolor=self.fg_color)
        style.map("TCombobox", 
                  fieldbackground=[("readonly", self.card_bg)],
                  foreground=[("readonly", self.fg_color)])

        # プログレスバーのスタイル定義
        style.configure("Horizontal.TProgressbar", thickness=15, troughcolor=self.card_bg, background=self.success_color, borderwidth=0)
        
    def create_widgets(self):
        """ウィジェットの配置とレイアウト構成"""
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. ヘッダーエリア
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_lbl = ttk.Label(header_frame, text="🍎 りんご画像検出システム", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)
        
        subtitle_lbl = ttk.Label(header_frame, text=" YOLOv11 / PyTorch パイプライン ", style="TLabel")
        subtitle_lbl.pack(side=tk.RIGHT, pady=(5, 0))

        # 区切り線
        divider = tk.Frame(main_container, height=2, bg=self.border_color)
        divider.pack(fill=tk.X, pady=(0, 15))

        # 2. 設定エリア（左：設定パネル、右：サマリーカード）
        config_and_stats = ttk.Frame(main_container)
        config_and_stats.pack(fill=tk.X, pady=(0, 15))
        
        # 左パネル: 入力設定
        inputs_frame = ttk.Frame(config_and_stats)
        inputs_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 対象画像フォルダの選択行
        folder_frame = ttk.Frame(inputs_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(folder_frame, text="対象画像フォルダ:").pack(anchor=tk.W, pady=(0, 2))
        
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.source_dir_var)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        folder_btn = ttk.Button(folder_frame, text="フォルダ選択", command=self.browse_folder)
        folder_btn.pack(side=tk.RIGHT)
        
        # YOLO重みファイルの選択行
        model_frame = ttk.Frame(inputs_frame)
        model_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(model_frame, text="YOLOモデル重みファイル (.pt):").pack(anchor=tk.W, pady=(0, 2))
        
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var)
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        model_btn = ttk.Button(model_frame, text="モデル選択", command=self.browse_model)
        model_btn.pack(side=tk.RIGHT)
        
        # パラメータ設定グリッド
        params_frame = ttk.Frame(inputs_frame)
        params_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 確信度しきい値スライダー
        conf_container = ttk.Frame(params_frame)
        conf_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.conf_lbl_var = tk.StringVar(value="確信度しきい値: 0.25")
        ttk.Label(conf_container, textvariable=self.conf_lbl_var).pack(anchor=tk.W)
        
        conf_scale = ttk.Scale(conf_container, from_=0.01, to=1.0, variable=self.conf_var, 
                               command=lambda v: self.conf_lbl_var.set(f"確信度しきい値: {float(v):.2f}"))
        conf_scale.pack(fill=tk.X, pady=2)
        
        # NMS IoUしきい値スライダー
        iou_container = ttk.Frame(params_frame)
        iou_container.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.iou_lbl_var = tk.StringVar(value="NMS IoUしきい値: 0.45")
        ttk.Label(iou_container, textvariable=self.iou_lbl_var).pack(anchor=tk.W)
        
        iou_scale = ttk.Scale(iou_container, from_=0.01, to=1.0, variable=self.iou_var,
                             command=lambda v: self.iou_lbl_var.set(f"NMS IoUしきい値: {float(v):.2f}"))
        iou_scale.pack(fill=tk.X, pady=2)

        # 実行デバイス＆画像サイズ
        dev_container = ttk.Frame(params_frame)
        dev_container.pack(side=tk.RIGHT, fill=tk.X, padx=(10, 0))
        
        ttk.Label(dev_container, text="デバイス:").pack(anchor=tk.W)
        device_choices = ["cpu", "cuda", "mps"]
        self.dev_combo = ttk.Combobox(dev_container, textvariable=self.device_var, values=device_choices, width=8, state="readonly")
        self.dev_combo.pack(pady=2)
        
        # 出力フォルダの選択行
        output_frame = ttk.Frame(inputs_frame)
        output_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(output_frame, text="出力フォルダ:").pack(anchor=tk.W, pady=(0, 2))
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        output_btn = ttk.Button(output_frame, text="フォルダ選択", command=self.browse_output)
        output_btn.pack(side=tk.RIGHT)
        
        # 右パネル: サマリー情報表示カード
        stats_card = ttk.Frame(config_and_stats, style="Card.TFrame", padding=15, width=220)
        stats_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        stats_card.pack_propagate(False)
        
        ttk.Label(stats_card, text="📊 処理結果サマリー", style="CardHeader.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        # 画像総数表示
        ttk.Label(stats_card, text="処理済み画像数", style="StatsLbl.TLabel").pack(anchor=tk.W)
        self.stat_images_val = ttk.Label(stats_card, text="0", style="StatsVal.TLabel")
        self.stat_images_val.pack(anchor=tk.W, pady=(0, 8))
        
        # 検出されたりんご総数表示
        ttk.Label(stats_card, text="検出りんご総数", style="StatsLbl.TLabel").pack(anchor=tk.W)
        self.stat_apples_val = ttk.Label(stats_card, text="0", style="StatsVal.TLabel")
        self.stat_apples_val.pack(anchor=tk.W, pady=(0, 8))
        
        # 平均検出個数表示
        ttk.Label(stats_card, text="画像あたりの平均個数", style="StatsLbl.TLabel").pack(anchor=tk.W)
        self.stat_avg_val = ttk.Label(stats_card, text="0.00", style="StatsVal.TLabel")
        self.stat_avg_val.pack(anchor=tk.W)

        # 3. ログコンソールエリア
        log_label_frame = ttk.Frame(main_container)
        log_label_frame.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(log_label_frame, text="ログ出力コンソール:").pack(side=tk.LEFT)
        
        self.log_text = scrolledtext.ScrolledText(main_container, bg="#181825", fg=self.fg_color, 
                                                 insertbackground=self.fg_color, font=("Consolas", 10), 
                                                 height=16, borderwidth=1, relief="solid")
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.log_text.bind("<Key>", lambda e: "break") # 読み取り専用に設定
        
        # 初期ログ
        self.append_log("システム準備完了。画像フォルダを選択し、「処理開始」をクリックしてください。\n")

        # 4. 実行ボタン＆プログレスバー
        actions_frame = ttk.Frame(main_container)
        actions_frame.pack(fill=tk.X, pady=(0, 5))
        
        # ステータスラベル
        self.progress_lbl_var = tk.StringVar(value="待機中")
        self.progress_lbl = ttk.Label(actions_frame, textvariable=self.progress_lbl_var, font=("Meiryo UI", 9, "italic"))
        self.progress_lbl.pack(side=tk.LEFT)
        
        # 右側ボタン配置
        self.open_dir_btn = ttk.Button(actions_frame, text="📁 出力フォルダを開く", style="Open.TButton", 
                                       command=self.open_output_folder, state=tk.DISABLED)
        self.open_dir_btn.pack(side=tk.RIGHT, padx=5)
        
        self.start_btn = ttk.Button(actions_frame, text="🚀 処理開始", style="Action.TButton", command=self.start_process)
        self.start_btn.pack(side=tk.RIGHT, padx=5)
        
        self.progress_bar = ttk.Progressbar(main_container, orient=tk.HORIZONTAL, mode='determinate', style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X)

    def browse_folder(self):
        dir_selected = filedialog.askdirectory(initialdir=self.source_dir_var.get(), title="画像フォルダを選択")
        if dir_selected:
            self.source_dir_var.set(os.path.normpath(dir_selected))

    def browse_model(self):
        file_selected = filedialog.askopenfilename(initialdir=os.path.dirname(self.model_path_var.get()),
                                                   title="YOLO重みファイルを選択",
                                                   filetypes=[("PyTorch重みファイル", "*.pt"), ("すべてのファイル", "*.*")])
        if file_selected:
            self.model_path_var.set(os.path.normpath(file_selected))

    def browse_output(self):
        dir_selected = filedialog.askdirectory(initialdir=self.output_dir_var.get(), title="出力フォルダを選択")
        if dir_selected:
            self.output_dir_var.set(os.path.normpath(dir_selected))

    def append_log(self, text):
        """コンソールへのログ追記"""
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def thread_safe_log(self, text):
        self.root.after(0, lambda: self.append_log(text + "\n"))

    def thread_safe_progress(self, idx, total):
        self.root.after(0, lambda: self.update_progress(idx, total))

    def update_progress(self, idx, total):
        pct = (idx / total) * 100
        self.progress_bar['value'] = pct
        self.progress_lbl_var.set(f"画像処理中: {idx} / {total}枚 ({pct:.1f}%)")

    def start_process(self):
        if self.is_running:
            return
        
        src = self.source_dir_var.get()
        model = self.model_path_var.get()
        out = self.output_dir_var.get()
        
        # 入力値の簡易バリデーション
        if not os.path.exists(src):
            messagebox.showerror("エラー", f"対象画像フォルダ '{src}' が存在しません。")
            return
        if not os.path.exists(model):
            messagebox.showerror("エラー", f"YOLOモデル重みファイル '{model}' が存在しません。")
            return
            
        # UIの操作無効化（ロック）
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.open_dir_btn.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        self.progress_lbl_var.set("検出パイプラインを初期化中...")
        
        # サマリー表示のリセット
        self.stat_images_val.config(text="-")
        self.stat_apples_val.config(text="-")
        self.stat_avg_val.config(text="-")

        self.log_text.delete('1.0', tk.END)
        self.append_log(f"▶️ りんご検出処理を開始しました: {src}\n")
        self.append_log(f"モデルパス    : {model}\n")
        self.append_log(f"出力フォルダ  : {out}\n")
        self.append_log("-" * 60 + "\n")

        # バックグラウンドスレッドの起動
        self.processing_thread = threading.Thread(target=self.run_pipeline, args=(src, model, out))
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def run_pipeline(self, src, model, out):
        try:
            # detect.pyの処理をコールバック付きで実行
            run_detection(
                source=src,
                model_path=model,
                conf_thresh=self.conf_var.get(),
                iou_thresh=self.iou_var.get(),
                imgsz=int(self.imgsz_var.get()),
                output_dir=out,
                device=self.device_var.get(),
                show=False,
                progress_callback=self.thread_safe_progress,
                log_callback=self.thread_safe_log
            )
            
            # 処理完了後のコールバックをUIスレッドで呼ぶ
            self.root.after(0, lambda: self.on_pipeline_success(src, out))
            
        except Exception as e:
            self.thread_safe_log(f"\n❌ パイプラインエラー: {str(e)}")
            self.root.after(0, lambda: self.on_pipeline_failure(str(e)))

    def on_pipeline_success(self, src, out):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.open_dir_btn.config(state=tk.NORMAL)
        self.progress_lbl_var.set("処理が正常に完了しました！")
        
        # ログコンソールから最終サマリーデータを抽出してカードを更新
        log_content = self.log_text.get('1.0', tk.END)
        total_img = "0"
        total_apples = "0"
        avg_apples = "0.00"
        
        for line in log_content.splitlines():
            if "Total Images Processed :" in line:
                total_img = line.split(":")[-1].strip()
            elif "Total Apples Detected  :" in line:
                total_apples = line.split(":")[-1].strip()
            elif "Average Apples / Image :" in line:
                avg_apples = line.split(":")[-1].strip()
                
        self.stat_images_val.config(text=total_img)
        self.stat_apples_val.config(text=total_apples)
        self.stat_avg_val.config(text=avg_apples)
        
        messagebox.showinfo("成功", "りんごの検出処理が正常に完了しました！")

    def on_pipeline_failure(self, error_msg):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.progress_lbl_var.set("画像検出中にエラーが発生しました。")
        messagebox.showerror("実行失敗", f"検出処理中にエラーが発生しました:\n\n{error_msg}")

    def open_output_folder(self):
        out_dir = self.output_dir_var.get()
        if os.path.exists(out_dir):
            try:
                os.startfile(out_dir)
            except Exception as e:
                # 代替のエクスプローラー起動
                import subprocess
                subprocess.run(["explorer", os.path.normpath(out_dir)])
        else:
            messagebox.showwarning("フォルダが見つかりません", f"出力フォルダ '{out_dir}' が存在しません。")


def main():
    root = tk.Tk()
    app = AppleDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
