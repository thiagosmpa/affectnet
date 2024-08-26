## AffectNet Emotion Classification

**Classifique emoções em vídeos com precisão utilizando modelos de Deep Learning.**

![output](output/output.gif)

Este projeto implementa localmente o modelo EMO-AffectNet, desenvolvido por @ElenaRyumina, para classificar emoções a partir de vídeos. A arquitetura combina redes neurais convolucionais (CNNs) para extração de características e Long Short-Term Memory (LSTM) para análise temporal, permitindo uma compreensão mais profunda das expressões faciais ao longo do tempo.

### Requisitos:

* Python 3.10
* Bibliotecas listadas em `requirements.txt`

### Download de Modelos:

* **Modelos Backbone:** [Baixe aqui](https://drive.google.com/drive/folders/1ahiKWj6gJ7yC2ye6vBEy0GJfdeguplq4)
* **Modelos LSTM:** [Baixe aqui](https://drive.google.com/drive/folders/1m7ATft4STye2Wiip3BZNUGIkducHC0SD)

### Funcionamento:

1. **Extração de Características:** As CNNs processam cada quadro do vídeo, extraindo características relevantes das expressões faciais.
2. **Análise Temporal:** A LSTM recebe as sequências de características e modela as dependências temporais, capturando a evolução das emoções ao longo do vídeo.
3. **Classificação:** A saída da LSTM é utilizada para classificar a emoção predominante no vídeo.

### Como Usar:

1. **Clone o Repositório:**
   ```bash
   git clone https://github.com/seu-usuario/Affectnet-Emotion-Classification.git
   ```

2. **Instale as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Baixe os Modelos:**
   * Faça o download dos modelos backbone e LSTM dos links fornecidos acima.
   * Coloque os modelos nas pastas apropriadas dentro do projeto.

4. **Execute o Script Principal:**
   ```bash
   python main.py --video_path caminho/para/seu/video.mp4
   ```

### Contribuições:

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

### Licença:

Este projeto está licenciado sob a [MIT License](LICENSE).

**Observações:**

* Certifique-se de ter os requisitos de hardware e software adequados para executar o projeto.
* Adapte o código para suas necessidades específicas, como alterar o caminho para o vídeo de entrada ou personalizar os modelos utilizados.
* Explore o código e os modelos para aprofundar seu conhecimento sobre classificação de emoções e deep learning.

**Com o AffectNet Emotion Classification, desvende as emoções por trás das expressões faciais em vídeos!** 
