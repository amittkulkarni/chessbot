import os
import argparse
from dotenv import load_dotenv
from src.search.mcts import ONNXEngine
from src.env.lichess_client import LichessClient
from src.model.export import export_and_quantize

def main():
    parser = argparse.ArgumentParser(description="Hardware-aware Deep Learning Chess Bot")
    parser.add_argument("--export", action="store_true", help="Export PyTorch model to ONNX Int8 before running")
    args = parser.parse_args()

    load_dotenv()

    API_TOKEN = os.getenv("LICHESS_API_TOKEN")
    PTH_PATH = os.getenv("PTH_WEIGHTS_PATH", "./weights/resnet20_student_v3.pth")
    ONNX_PATH = os.getenv("ONNX_MODEL_PATH", "./weights/resnet20_student_v3_int8.onnx")
    BOOK_PATH = os.getenv("BOOK_PATH", "./openings/Titans.bin")

    if args.export or not os.path.exists(ONNX_PATH):
        print("Running Post-Training Quantization to ONNX...")
        export_and_quantize(PTH_PATH, ONNX_PATH)

    print("Initializing Engine and Client...")
    engine = ONNXEngine(ONNX_PATH)

    bot = LichessClient(
        token=API_TOKEN,
        engine=engine,
        book_path=BOOK_PATH,
        max_think_time=20.0,
        c_puct=0.9
    )

    bot.start_listening()

if __name__ == "__main__":
    main()