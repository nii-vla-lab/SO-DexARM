# SO-DexARM

SO-DexARMは、Meta Quest 3Sによるハンドトラッキングを用いて、AmazingHand + SO-ARM101から構成されるDual-Armの遠隔操作、データ収集、学習、推論を行うための統合システムです。このリポジトリでは、一連の処理をHugging Face LeRobot上に実装しています。

## SO-DexARMの手順書

以下では、SO-DexARMの組み立て、環境構築、キャリブレーション、遠隔操作、データ収集、学習および推論までの手順を説明します。

### 1. 購入品一覧

SO-DexARMを組み立てるために必要な部品は、以下の部品表を参照してください。部品名、個数、参考価格、購入先URLなどを掲載しています。

[SO-DexARM部品表](https://1drv.ms/x/c/914548d97826a169/IQCNplq_oWRKTYGIElU2dYqKAYI0pj9Pnvb03ao_Z-FpBw8)

パーツの3Dプリントは以下のCADデータを使用してください。

- [`cad/AmazingHand/`](cad/AmazingHand/): AmazingHandのCADデータ
- [`cad/SO-ARM101/`](cad/SO-ARM101/): グリッパーとストッパーを除去したSO-ARM101のCADデータ

### 2. 環境構築

以下の環境構築の手順はUbuntu 22.04.5 LTSを想定しております。

#### 2.1 システムパッケージ

```bash
sudo apt update
sudo apt install -y git git-lfs adb
git lfs install
```

Minicondaをインストールするために以下のコマンドを実行してください。

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

インストールが完了したら、Terminatorを再起動してください。

#### 2.2 Python環境

このリポジトリのクローンおよびHand Tracking Streamerをサブモジュールとして取得するために以下のコマンドを実行してください。

```bash
git clone --recurse-submodules https://github.com/nii-vla-lab/SO-DexARM.git
cd SO-DexARM
```

Python 3.10の環境作成およびSO-DexARM用の依存ライブラリをインストールするために以下のコマンドを実行してください。

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install -y ffmpeg -c conda-forge
pip install -e ".[so-dexarm]"
```

### 3. 組み立て

#### 3.1 SO-ARM101

基本的な組み立て手順は公式の[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)を参照してください。

#### 3.2 AmazingHand

SO-ARM101同様に、基本的な組み立て手順は公式の[AmazingHand Assembly Guide](https://raw.githubusercontent.com/pollen-robotics/AmazingHand/main/docs/AmazingHand_Assembly.pdf)を参照してください。

#### 3.3 配線

実機とポート名の構成は以下のようにしてください。

| 実機 | ポート名 |
| --- | --- |
| Right SO-ARM101 | `/dev/ttyso101_amazinghand_r_arm` |
| Left SO-ARM101 | `/dev/ttyso101_amazinghand_l_arm` |
| Right AmazingHand | `/dev/ttyso101_amazinghand_r_hand` |
| Left AmazingHand | `/dev/ttyso101_amazinghand_l_hand` |

### 4. ポートの固定

udevルールでポート名を固定させるために以下のコマンドを実行してください。

```bash
cp scripts/udev/99-so-dexarm.rules.example scripts/udev/99-so-dexarm.rules
```

各シリアルバスサーボドライバーボードを1台ずつ接続、デバイス情報を確認するために以下のコマンドを実行してください。

```bash
lerobot-find-port
udevadm info --query=property --name=/dev/ttyACM0 \
  | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT'
```

`scripts/udev/99-so-dexarm.rules`内の`<...>`を、実際のVendor ID、Product IDおよびシリアル番号へ書き換えてください。

ルールを適用させるために、以下のコマンドを実行してください。

```bash
sudo cp scripts/udev/99-so-dexarm.rules /etc/udev/rules.d/99-so-dexarm.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 5. モーターIDとキャリブレーション

#### 5.1 SO-ARM101のモーターID

左右のアームのモーターIDを割り当てるために、以下のコマンドを実行してください。

```bash
./scripts/setup_motors.sh right
./scripts/setup_motors.sh left
```

デフォルトでは、モーターIDを1~5に割り当てるようになってますが、モーターIDを独自で割り当てたい場合はCLIオプションを追加して、以下のコマンドを実行してください。

```bash
./scripts/setup_motors.sh right \
  --right-arm-port /dev/ttyACM0 \
  --right-arm-ids 1,2,3,4,5
```

#### 5.2 SO-ARM101の可動域キャリブレーション

画面の指示に従って、各関節を最大可動域と最小可動域の中間くらいを維持した状態で、以下のコマンドを実行、稼働範囲全体を満遍なく動かして最大可動域および最小可動域を設定してください。

```bash
lerobot-so-dexarm calibrate-arm --side right
lerobot-so-dexarm calibrate-arm --side left
```

#### 5.3 AmazingHandとQuestランドマークの対応付け

Meta Quest 3SでHand Tracking Streamerをインストールおよび起動、TCP接続できる状態にしてから以下のコマンドを実行してください。各プロンプトで、Amazing Handと人間の手を同じ全開(`open`)、中間(`mid`)、全閉(`fist`)姿勢にしてください。

```bash
lerobot-so-dexarm calibrate-hand \
  --side right \
  --all-poses \
  --from-hardware

lerobot-so-dexarm calibrate-hand \
  --side left \
  --all-poses \
  --from-hardware
```

変換結果を確認したい場合は以下を実行してください。

```bash
lerobot-so-dexarm map-hand --side both
```

#### 5.4 開始姿勢の保存

データ収集時の開始姿勢の設定をするために、以下のコマンドを実行してください。

```bash
lerobot-so-dexarm capture-startup --side both --from-hardware
```

### 6. Meta Quest 3SとHand Tracking Streamer

SO-DexARMにおけるトラッキング機能は[wengmister/hand-tracking-streamer](https://github.com/wengmister/hand-tracking-streamer)を使用します。このリポジトリでは`./scripts/hand-tracking-streamer/`にサブモジュールとして配置しています。

SO-DexARMのQuest受信処理はTCPポート8000を使用するために、以下のコマンドを実行してください。

```bash
adb reverse tcp:8000 tcp:8000
adb reverse --list
```

Hand Tracking Streamer側の設定は以下のようにしてください。

```text
Protocol: TCP
Host: localhost
Port: 8000
Hand side: Both Hands
```

無線TCPを使用する場合、QuestとPCを同じネットワークへ接続し、`Host`にPCのIPv4アドレス、`Port`に`8000`を指定してください。

### 7. 遠隔操作

テレオペするために以下のコマンドを実行してください。

```bash
./scripts/teleop.sh
```

このシステムのアーム制御は、`shoulder_pan`、`shoulder_lift`、`elbow_flex`および`wrist_flex`を使用する拘束付き平面IK、`wrist_roll`は遠隔操作開始時の角度に固定されるようになっています。

### 8. カメラ設定

RealSenseを接続して、以下のコマンドを実行してください。

```bash
lerobot-find-cameras realsense
```

確認したシリアル番号を使って、`<CAMERA_SERIAL>`を環境変数へ保存するために以下のコマンドを実行してください。

```bash
export CAMERAS='{cam_front: {type: intelrealsense, serial_number_or_name: <CAMERA_SERIAL>, width: 640, height: 480, fps: 30}}'
```

### 9. データ収集

タスク指示やカメラ設定などを書き換えて、以下のコマンドを実行してください。

```bash
export TASK='Fold the towel.'
export CAMERAS='{cam_front: {type: intelrealsense, serial_number_or_name: <CAMERA_SERIAL>, width: 640, height: 480, fps: 30}}'

NUM_EPISODES=50 \
EPISODE_TIME_S=30 \
RESET_TIME_S=5 \
./scripts/record.sh '<HUGGINGFACE_USER>/<DATASET_NAME>' --push-to-hub
```

#### 9.1 データセットの編集

失敗したエピソードを削除して別データセットへ保存する場合は以下のコマンドを実行してください。

```bash
./scripts/edit.sh \
  --repo_id '<HUGGINGFACE_USER>/<DATASET_NAME>' \
  --new_repo_id '<HUGGINGFACE_USER>/<FILTERED_DATASET_NAME>' \
  --operation.type delete_episodes \
  --operation.episode_indices '[0, 3, 7]'
```

### 10. 学習

#### 10.1 ACT

```bash
POLICY_DEVICE=cuda \
STEPS=100000 \
BATCH_SIZE=16 \
./scripts/train_act.sh '<HUGGINGFACE_USER>/<DATASET_NAME>'
```

#### 10.2 SmolVLA

```bash
POLICY_DEVICE=cuda \
STEPS=20000 \
BATCH_SIZE=8 \
./scripts/train_smolvla.sh '<HUGGINGFACE_USER>/<DATASET_NAME>'
```

### 11. 実機推論

データ収集時のカメラ設定と同じにして、以下のコマンドを実行してください。

```bash
export CAMERAS='{cam_front: {type: intelrealsense, serial_number_or_name: <CAMERA_SERIAL>, width: 640, height: 480, fps: 30}}'
```

#### 11.1 ACT

```bash
./scripts/eval_act.sh \
  'outputs/train/so-dexarm-act-<DATASET_NAME>/checkpoints/last/pretrained_model'
```

#### 11.2 SmolVLA

SmolVLAでは、データ収集時と同じタスク文に書き換えてください。

```bash
TASK='Fold the towel.' \
./scripts/eval_smolvla.sh \
  'outputs/train/so-dexarm-smolvla-<DATASET_NAME>/checkpoints/last/pretrained_model'
```

評価データセット名を変更する場合は、`/`以降を`eval_`で開始してください。

```bash
EVAL_REPO_ID=local/eval_towel \
NUM_EPISODES=5 \
./scripts/eval_act.sh '<PRETRAINED_POLICY_PATH>'
```

## Upstream projects

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [TheRobotStudio SO-ARM100/SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100)
- [Pollen Robotics AmazingHand](https://github.com/pollen-robotics/AmazingHand)
- [Hand Tracking Streamer](https://github.com/wengmister/hand-tracking-streamer)
