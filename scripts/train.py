import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.resnet import ChessResNet
from src.utils.data_loader import ChessDataset18

def train_model():
    CONFIG = {
        "num_features": 18, "num_moves": 4096, "num_res_blocks": 20, "num_channels": 256,
        "batch_size": 2048, "num_epochs": 3, "lr": 0.001, "max_train_hours": 12.0,
        "data_dir": "./data/chess-stockfish-data",
        "save_dir": "./weights/",
        "resume_from": None
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count()
    session_start_time = time.time()

    model = ChessResNet(CONFIG["num_features"], CONFIG["num_moves"],
                        CONFIG["num_res_blocks"], CONFIG["num_channels"], num_value_targets=1)
    model = model.to(DEVICE)

    if NUM_GPUS > 1:
        print(f"⚡ Enabling PyTorch DataParallel on {NUM_GPUS} GPUs")
        model = nn.DataParallel(model)

    start_epoch = 0
    if CONFIG["resume_from"] and os.path.exists(CONFIG["resume_from"]):
        checkpoint = torch.load(CONFIG["resume_from"], map_location=DEVICE)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    dataset = ChessDataset18(CONFIG["data_dir"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], num_workers=4, pin_memory=True)

    model.train()

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        total_p_loss, total_v_loss = 0, 0
        batch_count = 0

        for boards, move_ids, scores in dataloader:
            boards = boards.to(DEVICE)
            move_ids = move_ids.long().to(DEVICE)
            scores = scores.float().to(DEVICE).view(-1, 1)

            optimizer.zero_grad()
            p_out, v_out = model(boards)

            loss_p = criterion_p(p_out, move_ids)
            # Apply sigmoid to value head if predicting WDL as single float
            loss_v = criterion_v(torch.sigmoid(v_out), scores)
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            batch_count += 1

            if batch_count % 100 == 0:
                print(f"E{epoch+1}|B{batch_count} >> P:{loss_p.item():.3f} | V:{loss_v.item():.3f}")

        scheduler.step()

        save_name = f"resnet20_epoch{epoch+1}.pth"
        save_path = os.path.join(CONFIG["save_dir"], save_name)
        os.makedirs(CONFIG["save_dir"], exist_ok=True)

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_model()