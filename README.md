# 🏍️ Rastreamento e Contagem de Motos com YOLOv8 + ByteTrack

Este projeto implementa um sistema de **detecção, rastreamento e contagem de motocicletas em tempo real**.  
O modelo utiliza [YOLOv8](https://github.com/ultralytics/ultralytics) para detecção e [ByteTrack](https://github.com/ifzhang/ByteTrack) para rastreamento de múltiplos objetos, com visualização das caixas, IDs, trilhas e métricas na tela.

---

## ✨ Funcionalidades
- 🚦 **Detecção em tempo real** de múltiplas motos em vídeo ou webcam.
- 🆔 **Rastreamento persistente** com IDs únicos por moto.
- 🔢 **Contagem acumulada** de todas as motos que apareceram no vídeo.
- 📊 **Métricas em HUD**: FPS, acurácia (confiança), número de objetos no frame e total acumulado.
- 📝 **Exportação opcional em CSV** com logs (timestamp, frame, track_id, bbox, confiança).
- 💾 **Gravação de vídeo processado** com detecções sobrepostas.

---

## 📦 Instalação

Crie um ambiente virtual (recomendado) e instale as dependências:

```bash
# criar ambiente virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# atualizar pip
pip install --upgrade pip

# instalar dependências principais
pip install ultralytics supervision opencv-python
```

Se precisar do PyTorch CPU/GPU manualmente, consulte: [Instruções oficiais](https://pytorch.org/get-started/locally/).

---

## ▶️ Uso

Rodar o script com um vídeo:

```bash
python track_motos.py --source test.mp4 --show-fps
```

Rodar usando webcam:

```bash
python track_motos.py --source 0 --show-fps
```

Gravar a saída anotada em vídeo:

```bash
python track_motos.py --source test.mp4 --save saida.mp4
```

Exportar log CSV com detecções:

```bash
python track_motos.py --source test.mp4 --export-csv motos_log.csv
```

Loopar vídeo até apertar `q`:

```bash
python track_motos.py --source test.mp4 --loop
```

---

## ⚙️ Argumentos Disponíveis

| Argumento         | Descrição                                                                 | Padrão      |
|-------------------|---------------------------------------------------------------------------|-------------|
| `--source`        | Fonte do vídeo (`0` para webcam ou caminho do arquivo)                    | `0`         |
| `--model`         | Modelo YOLOv8 a ser usado (ex.: `yolov8n.pt`, `yolov8s.pt`)               | `yolov8n.pt`|
| `--conf`          | Confiança mínima para detecção (0–1)                                      | `0.35`      |
| `--save`          | Caminho para salvar o vídeo de saída (`.mp4`)                             | _vazio_     |
| `--show-fps`      | Exibe FPS no HUD                                                          | `False`     |
| `--loop`          | Reinicia o vídeo automaticamente ao chegar no fim                        | `False`     |
| `--export-csv`    | Exporta log CSV com timestamp, frame, track_id, bbox e confiança          | _vazio_     |

---

## 📊 Saída

### Na tela
- Caixa colorida ao redor de cada moto detectada.
- Label com **ID + confiança** (`Moto 3 0.87`).
- Linha de rastro do movimento.
- HUD com FPS, acurácia, número de objetos no frame e total acumulado.

### No terminal
Ao final da execução:
```
================ RESULTADO ================
Total de motos que apareceram (IDs únicos): 42
==========================================
```

### Em CSV (opcional)
Cada linha contém:
```
timestamp_s, frame_idx, track_id, x1, y1, x2, y2, conf
```

---

## 📹 Vídeos para Teste
Você pode baixar vídeos gratuitos para teste em:
- [Pixabay – Motorcycle Videos](https://pixabay.com/videos/search/motorcycle/)
- [Pexels – Motorcycle Clips](https://www.pexels.com/search/videos/motorcycle/)
- [Mixkit – Free Motorcycle Footage](https://mixkit.co/free-stock-video/motorcycle/)

---

## 🛠️ Próximos Passos (Ideias)
- 🚧 Adicionar **linha virtual de contagem (in/out)** para monitorar fluxo direcional.
- 📡 Integração com API/web para monitoramento em tempo real.
- 📱 Aplicativo mobile para visualização das estatísticas.

---

## 👨‍💻 Autores
Projeto desenvolvido para fins acadêmicos (FIAP / Global Solution).  

---

## 📄 Licença
Este projeto é open-source sob a licença MIT.
