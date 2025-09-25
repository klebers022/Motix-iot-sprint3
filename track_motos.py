import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from collections import deque
from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


def parse_args():
    ap = argparse.ArgumentParser(
        description="Detecção, rastreamento e contagem de motos (YOLOv8 + ByteTrack)."
    )
    ap.add_argument("--source", type=str, default="0",
                    help="0 para webcam, ou caminho do vídeo (ex.: ./video.mp4)")
    ap.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Modelo YOLO (ex.: yolov8n.pt, yolov8s.pt, yolov8m.pt).")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Confidence mínima para detecção (0-1).")
    ap.add_argument("--save", type=str, default="",
                    help="Caminho do arquivo de saída .mp4 (opcional).")
    ap.add_argument("--show-fps", action="store_true",
                    help="Exibe FPS na imagem.")
    ap.add_argument("--loop", action="store_true",
                    help="Reinicia o vídeo automaticamente ao chegar no fim.")
    ap.add_argument("--export-csv", type=str, default="",
                    help="Exporta logs (timestamp, frame, track_id, bbox, conf) para CSV.")
    return ap.parse_args()


def id_color(track_id: int):
    np.random.seed(track_id % (2**32 - 1))
    return tuple(int(x) for x in np.random.randint(60, 255, size=3))


def draw_hud(frame, fps_val, num_now, num_unique, acc_frame, acc_avg):
    """HUD com métricas + total acumulado de motos (IDs únicos)."""
    pad = 10
    line_h = 24
    items = [
        (f"FPS: {fps_val:.1f}" if fps_val is not None else None),
        f"Objetos no frame: {num_now}",
        f"IDs únicos (Total acumulado): {num_unique}",
        (f"Acurácia (frame): {acc_frame:.2f}" if acc_frame is not None else "Acurácia (frame): --"),
        (f"Acurácia (média): {acc_avg:.2f}" if acc_avg is not None else "Acurácia (média): --"),
    ]
    items = [x for x in items if x is not None]

    box_w = 380
    box_h = pad * 2 + line_h * len(items)

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    y = pad + 18
    for txt in items:
        cv2.putText(frame, txt, (pad + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    return frame


def main():
    args = parse_args()

    source = 0 if args.source.strip() == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    model = YOLO(args.model)  # COCO: inclui 'motorcycle'
    tracker = sv.ByteTrack()
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=20)

    # IDs da classe "motorcycle"
    names = model.model.names if hasattr(model.model, "names") else model.names
    motorcycle_ids = [i for i, n in names.items() if n.lower() in ("motorcycle", "motorbike")]
    if not motorcycle_ids:
        raise RuntimeError("Classe 'motorcycle' não encontrada no modelo escolhido.")

    # Escrita de vídeo (opcional)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_src, (w, h))

    # CSV (opcional)
    csv_file = None
    csv_writer = None
    if args.export_csv:
        csv_path = Path(args.export_csv)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_s", "frame_idx", "track_id", "x1", "y1", "x2", "y2", "conf"])

    # Métricas
    unique_ids = set()           # <-- total de motos que já apareceram (IDs únicos)
    conf_window = deque(maxlen=100)
    last_time = time.time()
    fps = None
    frame_idx = 0

    window_title = "Rastreamento e Contagem de Motos - YOLOv8 + ByteTrack"

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_idx += 1

        # Inferência YOLO
        result = model(frame, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filtra motos
        if detections.class_id is not None and len(detections) > 0:
            mask = np.isin(detections.class_id, motorcycle_ids)
            detections = detections[mask]

        # Rastreamento (atribui/atualiza IDs)
        detections = tracker.update_with_detections(detections)

        # Labels, cores e métricas de confiança
        labels, colors, frame_conf_list = [], [], []
        for i in range(len(detections)):
            tid = int(detections.tracker_id[i]) if (
                detections.tracker_id is not None and detections.tracker_id[i] is not None
            ) else None
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            frame_conf_list.append(conf)

            if tid is not None:
                unique_ids.add(tid)  # <-- conta motos únicas
                labels.append(f"Moto {tid} {conf:.2f}")
                colors.append(id_color(tid))
            else:
                labels.append(f"Moto {conf:.2f}")
                colors.append(id_color(i))

        # atualiza médias de confiança
        acc_frame = float(np.mean(frame_conf_list)) if frame_conf_list else None
        if acc_frame is not None:
            conf_window.append(acc_frame)
        acc_avg = float(np.mean(conf_window)) if len(conf_window) else None

        # trilhas
        frame = trace_annotator.annotate(scene=frame, detections=detections)

        # preparar vetores seguros para zip (corrigido para ndarray)
        confidences = detections.confidence if detections.confidence is not None else np.zeros(len(detections))
        track_ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)

        # caixas + labels
        if len(detections) > 0:
            for (xyxy, label, clr, conf_val, tid) in zip(
                detections.xyxy, labels, colors, confidences, track_ids
            ):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_text = max(0, y1 - th - 6)
                cv2.rectangle(frame, (x1, y_text), (x1 + tw + 6, y_text + 6 + th), clr, -1)
                cv2.putText(frame, label, (x1 + 3, y_text + th + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # CSV log
                if csv_writer is not None:
                    timestamp_s = frame_idx / (fps_src if fps_src > 0 else 25.0)
                    csv_writer.writerow([
                        f"{timestamp_s:.3f}", frame_idx,
                        (int(tid) if tid is not None else -1),
                        x1, y1, x2, y2,
                        float(conf_val) if conf_val is not None else 0.0
                    ])

        # FPS
        if args.show_fps:
            now = time.time()
            inst = 1.0 / max(1e-6, (now - last_time))
            fps = 0.9 * (fps or inst) + 0.1 * inst
            last_time = now

        # HUD (inclui Total acumulado = len(unique_ids))
        frame = draw_hud(
            frame,
            fps_val=fps if args.show_fps else None,
            num_now=len(detections),
            num_unique=len(unique_ids),
            acc_frame=acc_frame,
            acc_avg=acc_avg
        )

        # exibir / gravar
        cv2.imshow(window_title, frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalização
    cap.release()
    if writer is not None:
        writer.release()
    if csv_file is not None:
        csv_file.close()

    cv2.destroyAllWindows()

    # ---------- RESULTADO FINAL ----------
    total_motos = len(unique_ids)
    print("\n================ RESULTADO ================")
    print(f"Total de motos que apareceram (IDs únicos): {total_motos}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
