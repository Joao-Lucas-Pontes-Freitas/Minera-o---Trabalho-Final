import gdown
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Baixa um arquivo CSV do Google Drive.")
parser.add_argument(
    "--file_id",
    type=str,
    default="180hGFlu-bf7h6oGDBV3UCVAPRWU3jH7l",
    help="ID do arquivo no Google Drive. O ID padrão é referente ao datase `Credit Card Fraud Detection` do kaggle.",
)
parser.add_argument(
    "--output_name",
    type=str,
    default="creditcardfraud_modified.csv",
    help="Nome do arquivo CSV de saída. O nome padrão é referente ao mesmo dataset do kaggle.",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default="./datasets/",
    help="Diretório de destino para salvar o arquivo",
)

args = parser.parse_args()

url = f"https://drive.google.com/uc?export=download&id={args.file_id}"

args.output_dir.mkdir(parents=True, exist_ok=True)

output_path = args.output_dir / args.output_name

gdown.download(url, str(output_path), quiet=False)

print(f"Arquivo CSV baixado para: {output_path}")
