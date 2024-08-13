import torch
import mediapipe as mp
import cv2
import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import annotate, get_box, display_FPS, pth_processing

def process_video(video_file, backbone_model, lstm_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp_face_mesh = mp.solutions.face_mesh

    # Model info
    pth_backbone_model = torch.jit.load(f'model/Torch/torchscript_model_{backbone_model}.pth').to(device)
    pth_backbone_model.eval()

    pth_LSTM_model = torch.jit.load(f'model/Torch/{lstm_model}.pth').to(device)
    pth_LSTM_model.eval()

    DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
    emotion_probs = {emotion: [] for emotion in DICT_EMO.values()}

    # Video info
    cap = cv2.VideoCapture(video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = np.round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video output
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = os.path.basename(video_file).split('.')[0]
    path_combined_video = f'{output_dir}/{backbone_model}_{lstm_model}_{video_name}.mp4'
    combined_title = f'Backbone: {backbone_model}, LSTM: {lstm_model}'
    combined_writer = cv2.VideoWriter(path_combined_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h + int(h/2)))

    lstm_features = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    plot_title = 'Emotions Probabilities'

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        for _ in tqdm(range(total_frames), desc="Processing frames"):
            t1 = time.time()
            success, frame = cap.read()
            if frame is None:
                break

            frame_copy = frame.copy()
            frame_copy.flags.writeable = False
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_copy)
            frame_copy.flags.writeable = True

            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    startX, startY, endX, endY = get_box(fl, w, h)
                    cur_face = frame_copy[startY:endY, startX:endX]

                    cur_face = pth_processing(Image.fromarray(cur_face))
                    features = torch.nn.functional.relu(pth_backbone_model.extract_features(cur_face)).cpu().detach().numpy()

                    if len(lstm_features) == 0:
                        lstm_features = [features] * 10
                    else:
                        lstm_features = lstm_features[1:] + [features]

                    lstm_f = torch.from_numpy(np.vstack(lstm_features))
                    lstm_f = torch.unsqueeze(lstm_f, 0).to(device)
                    output = pth_LSTM_model(lstm_f).cpu().detach().numpy()

                    for i, emotion in DICT_EMO.items():
                        emotion_probs[emotion].append(output[0, i])

                    cl = np.argmax(output)
                    label = DICT_EMO[cl]
                    frame = annotate(frame, (startX, startY, endX, endY), label, title=combined_title)

            t2 = time.time()

            frame = display_FPS(frame, 'FPS: {0:.1f}'.format(1 / (t2 - t1)), box_scale=.5)

            ax.clear()
            for emotion, probs in emotion_probs.items():
                ax.plot(probs, label=emotion)
                if probs:
                    ax.annotate(emotion, 
                                xy=(len(probs) - 1, probs[-1]), 
                                xytext=(5, 0), 
                                textcoords='offset points',
                                color=ax.get_lines()[-1].get_color(),
                                fontsize=10,
                                fontweight='regular')

            ax.legend()
            ax.set_xlabel('Frame')
            ax.set_ylabel('Probability')
            ax.set_title(plot_title)
            ax.grid(True)

            fig.canvas.draw()

            plot_frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            plot_frame = plot_frame.reshape(int(fig.get_size_inches()[1]*fig.get_dpi()), 
                                            int(fig.get_size_inches()[0]*fig.get_dpi()), 4)
            plot_frame = cv2.cvtColor(plot_frame, cv2.COLOR_RGBA2BGR)

            plot_frame = cv2.resize(plot_frame, (w, int(h/2)))

            combined_frame = np.vstack((frame, plot_frame))
            combined_writer.write(combined_frame)

    cap.release()
    combined_writer.release()

    plt.close(fig)

    return path_combined_video