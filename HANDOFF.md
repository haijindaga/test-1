# プロジェクト引き継ぎ書 — Safe LLM-Driven Robotics（Cross-Layer 再現＋拡張）

このドキュメントは Claude Code に渡す前提のまとめです。論文PDFは別途貼ります。

---

## 0. ゴール

**「Cross-Layer Sequence Supervision Mechanism」(IROS 2024) のフレームワークを、
GPT-4 APIではなくローカルLLMで再現し、最終的に独自の安全レイヤーを足して論文化する。**

- 主ターゲット論文（コード無し）: *Ensuring Safety in LLM-Driven Robotics: A Cross-Layer
  Sequence Supervision Mechanism*, Wang et al., IROS 2024.
- 参考＆コード流用元（MITライセンス・コードあり）: *RoboGuard: Safety Guardrails for
  LLM-enabled Robots*, Ravichandran et al., arXiv:2503.07885.
  - 依存ライブラリ **spot は GPLv3**（再配布時のみ注意。研究利用は問題なし）。
  - 論文を使う/論文化する場合は **参考文献で両論文を必ず引用**（READMEだけでは学術的に不十分）。

---

## 1. 核心アーキテクチャの理解（前提知識）

- **LTL（線形時相論理）** = 安全ルールを時間付き論理式で書く。例 `G(!grab_salmon | F open_microwave)`。
- **Büchi/NBA オートマトン** = LTL式を「判定機械（状態機械）」にコンパイルしたもの。
  正規表現→NFAの関係と同じ。**`spot` ライブラリ**が変換と判定を担う。
- **安全保証はLLMではなく spot（形式手法）が出す。** だからLLMをローカルの小型モデルに
  替えても検証の厳密さは落ちない。これが両論文＆本プロジェクトの肝。

### Cross-Layer の構成（= 本命）
- **LLMは1つだけ**（タスクプランナー。行動の生成と再生成）。
- **safety supervisor = NBA（非LLM・純アルゴリズム）。LTL制約は人間が事前定義。**
- 2つのレイヤー:
  1. **task層 / CheckSafety (Alg.1)**: 各行動がLTL制約に違反するか判定し、違反なら
     行動を破棄→「どの制約を破ったか」をLLMにフィードバックして再生成（閉ループ修正）。
  2. **motion層 / FindObstacle (Alg.2)**: LTLで禁止された領域を「障害物」として
     モーションプランナーに注入し、軌道がそこを避けるようにする。
- 論文実験: GPT-4.0 / Ubuntu20.04 / ROS Noetic / Gazebo / Franka arm・TurtleBot。

### RoboGuard との違い（流用ポイント）
- RoboGuard は **LLMがシーングラフからLTLを自動生成**する点が追加（Cross-Layerは手書き固定）。
- RoboGuard の `synthesis.py::ControlSynthesis.validate_action_sequence` が
  「行動列をオートマトンに通して受理判定」を実装済み → **CheckSafetyの判定エンジンに流用可能**。

---

## 2. 実行環境・ハード（重要な制約）

- **Linux機: Ubuntu 22.04 / ROS2 Humble / MoveIt2 利用可 / GPU = RTX 4060 (VRAM 8GB) / RAM 32GB。**
- ローカルLLMは **Ollama**（OpenAI互換エンドポイント `http://localhost:11434/v1`）。
- モデル選定の結論:
  - `gemma3:27b` … 非thinkingで品質良。ただし8GB VRAMに載り切らずCPUへ溢れる（遅め）。
  - `qwen3.5:9b` … 8GBに完全に載って高速。task計画の短い出力なら十分。
  - `gemma3:12b` … 中間。
  - **注意: thinking系（qwen3.6等）は冗長で遅くなりがち。** 計画用途は非thinking推奨。
  - **VRAM競合**: Ollama と シミュレータ描画（Gazebo/CoppeliaSim/RViz）が8GBを取り合う。
    → 計画フェーズ（LLM）と実行フェーズ（sim）を**時間分割**すれば回避できる。
- LLM差し替えパターン（全箇所共通）:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
  client.chat.completions.create(model="gemma3:27b", ...)
  ```

---

## 3. 既に作ったもの（このリポジトリに含める）

- `task_layer.py` … **task層の動くスケルトン（sim非依存・純Python）**。
  - `SafetySupervisor`（NBA群 + Algorithm 1 `check_safety`）
  - `LLMPlanner`（Ollama経由の生成/再生成）
  - `SymbolicEnv`（記号的な状態遷移 T。後でRLBench/MoveItに差し替え）
  - `run_closed_loop`（Fig.2のループ。安全な時だけ状態更新）
- `generator.py` … RoboGuardの `generator.py` を**ローカルLLM + spot検証リトライ**に差し替えた版
  （RoboGuard側を試す場合のドロップイン）。

### 既知の要注意点（Claude Codeへ申し送り）
- **CheckSafetyの意味論**: 現状は「最終状態が受理状態か」で近似。`G/!`系は妥当だが、
  `X`(next)/`U`(until) を含む制約では prefix（有限列）の安全性判定を厳密化する必要あり。
- **AP（原子命題）モデル**: 「1ステップ＝その行動だけTrue、他は全部False」を仮定。
- **spot のインストールは pip 不可**（PyPIの `spot` は無関係な別物）。
  → `conda install -c conda-forge spot` か Ubuntu の apt（spotの公式リポジトリ）を使う。

---

## 4. 次にやること（推奨順）

1. **(c) `SafetySupervisor.check_safety` のユニットテスト**（Ollama不要・spotだけで検証）。
   危険な行動列がちゃんとSFeedbackで弾かれることを固める。
2. **(b) Cross-Layer 論文 Table II のシナリオを再現**（「色ごとに仕分け」等）。
   行動空間 `A` と `SymbolicEnv` の遷移/終了条件を具体化し、**安全率（safety rate）を再現**。
3. **motion層**を足す。simの選択は下記の2択（task層コードはどちらでも共通）:
   - **RLBench**（CoppeliaSim 4.1.0 + PyRep / Franka Panda標準 / 純Python / 経路計画内蔵）。
     禁止領域を **collision Shape** として置けば `arm.get_path` が避ける = FindObstacle相当。
     注意: CoppeliaSim 4.1.0 は元々Ubuntu20.04ビルド。22.04での動作要確認。
   - **ROS2 Humble + MoveIt2**（既存環境）。禁止領域を **PlanningScene の collision object** で
     注入、`moveit_py` でPythonから駆動。実機寄りだが接着コードは多め。
4. **(将来) 独自の第3レイヤー**を提案（新規性）。候補:
   動的制約の更新層 / 違反の重大度ランキング層 / 説明可能性（なぜ止めたか自然言語）層。

---

## 5. Git に上げて Linux で clone→実行（手順）

### 5-1. リポジトリを作って push（作業マシン側、初回のみ）

```bash
# プロジェクトフォルダで
cd /path/to/project
git init
git add task_layer.py generator.py HANDOFF.md .gitignore requirements.txt
git commit -m "Initial commit: task layer + handoff"

# GitHubに空リポジトリを作成（ブラウザ or gh CLI）
#  - ブラウザ: github.com で New repository（READMEは付けない）
#  - gh CLI:   gh repo create <name> --private --source=. --remote=origin --push
# 手動でremoteを繋ぐ場合:
git branch -M main
git remote add origin git@github.com:<user>/<repo>.git   # or https://...
git push -u origin main
```

### 5-2. Linux側で clone して環境構築・実行

```bash
git clone git@github.com:<user>/<repo>.git
cd <repo>

# 1) Python環境（conda推奨 = spotがcondaで入るため）
conda create -n safe-robot python=3.11 -y
conda activate safe-robot
conda install -c conda-forge spot -y          # ← LTL/automata（pip不可）
pip install -r requirements.txt               # openai など

# 2) Ollama（未導入なら）
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:27b        # or qwen3.5:9b（8GBで軽快）
ollama serve &                # バックグラウンドでサーバ起動

# 3) 実行
python task_layer.py
```

### 5-3. 以降の更新サイクル
```bash
# 作業マシンで編集後
git add -A && git commit -m "..." && git push
# Linux側で取り込み
git pull
```

> Claude Code に頼むなら: 「このHANDOFF.mdに沿って、まず §4 の(c)ユニットテストを書いて、
> 次に(b)のTable IIシナリオ用の SymbolicEnv と行動空間を実装して」と指示すればよい。
> §5 の git 初期化〜push も Claude Code に任せて問題ない（難易度低）。

---

## 6. requirements.txt（同梱）

```
openai>=1.0
# spot は conda-forge / apt で別途インストール（pip不可）
```
