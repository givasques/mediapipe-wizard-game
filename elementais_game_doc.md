# Elementais: Jogo de Feitiços com Visão Computacional

## 1. Visão Geral
Este projeto implementa um jogo 2D controlado por gestos das mãos, usando visão computacional em tempo real com MediaPipe.  
O jogador assume o papel de um mago e precisa derrotar inimigos elementais (fogo, gelo e elétrico) escolhendo o feitiço correto e apontando para o alvo com a mão oposta.

O trabalho foi desenvolvido com foco acadêmico para demonstrar, de forma prática, a integração entre:
- Processamento de imagem em tempo real;
- Reconhecimento de mãos e landmarks;
- Interpretação de gestos;
- Lógica de jogo baseada em regras.

## 2. Objetivo do Projeto
Construir um sistema interativo em que o controle tradicional (teclado/mouse) é substituído por gestos manuais, mantendo jogabilidade clara, feedback visual e progressão de dificuldade.

## 3. Tecnologias Utilizadas
- Python 3.10+;
- OpenCV (`cv2`) para captura de webcam, renderização e interface;
- MediaPipe Tasks Vision (Hand Landmarker) para detecção das mãos;
- NumPy para operações de imagem e sprites.

## 4. Estrutura do Repositório
```
mediapipe-wizard-game/
├─ elementais_game.py           # Loop principal do jogo, HUD, combate e telas
├─ game_logic.py           # Classificação de gestos e direção da mira
├─ vision.py               # Webcam, modelo MediaPipe e desenho de landmarks
├─ high_score.txt          # Persistência do recorde
├─ hand_landmarker.task    # Modelo do MediaPipe
├─ images/                 # Sprites e assets visuais
└─ specs/                  # Especificações conceituais do projeto
```

## 5. Arquitetura e Responsabilidades dos Módulos

### `vision.py`
Responsável pela camada de visão computacional:
- Abre e valida a webcam;
- Garante a presença do modelo `hand_landmarker.task` (download automático se necessário);
- Cria o detector de mãos;
- Converte frames BGR para formato do MediaPipe;
- Desenha landmarks e rótulos de mão na janela da webcam.

### `game_logic.py`
Responsável pela interpretação dos landmarks:
- Identifica lado da mão considerando imagem espelhada da câmera;
- Detecta se dedos estão levantados/abaixados com margem de tolerância;
- Classifica gesto da mão direita (ataque, defesa ou neutro);
- Classifica zona de mira da mão esquerda com base no ângulo do indicador.

### `elementais_game.py`
Responsável pela lógica de jogo e renderização:
- Inicialização das janelas, assets e estado global;
- Sistema de pontuação, vidas e dificuldade progressiva;
- Spawn de inimigos e projéteis;
- Colisão, dano, feedback textual e efeitos visuais;
- Telas de início, partida e game over;
- Persistência de recorde em arquivo local.

## 6. Mecânica de Controle por Gestos

### 6.1 Mão esquerda (mira)
A direção do indicador define uma zona de alvo:
- `1/3`: grupo da esquerda;
- `2/3`: grupo central;
- `3/3`: grupo da direita.

Essa zona é convertida em índice de inimigo-alvo no topo da tela.

### 6.2 Mão direita (ação)
Gestos implementados:
- Mão aberta (todos os dedos estendidos): **escudo** (`DEFESA`);
- Indicador levantado: **FOGO**;
- Indicador + médio (sinal de paz): **ÁGUA**;
- Polegar lateral + mindinho: **TERRA**;
- Outros padrões: estado **NEUTRO**.

O disparo é controlado por um mecanismo de anti-spam (`cast_armed`):  
o sistema só permite novo disparo após retorno ao estado neutro.

## 7. Regras de Combate

### 7.1 Fraquezas elementais
- Inimigo `FOGO` é derrotado por `ÁGUA`;
- Inimigo `GELO` é derrotado por `FOGO`;
- Inimigo `ELETRICO` é derrotado por `TERRA`.

### 7.2 Sistema de pontuação e vida
- Acerto correto: `+15` pontos;
- Tipo errado: `-10` pontos (mínimo 0);
- Cada inimigo possui 2 vidas;
- Jogador inicia com 5 vidas;
- Sem escudo, projétil inimigo reduz 1 vida.

### 7.3 Progressão de dificuldade
A dificuldade cresce conforme inimigos derrotados:
- Nível 1: padrão inicial;
- Nível 2: após 10 derrotas;
- Nível 3: após 20 derrotas;
- Nível 4: após 30 derrotas.

Com o aumento de nível, o jogo acelera:
- frequência de ataques inimigos;
- velocidade de projéteis;
- velocidade de entrada dos inimigos na cena.

## 8. Interface e Feedback ao Jogador
O sistema abre duas janelas:
- `Jogo`: cenário principal com player, inimigos, projéteis, HUD, vidas e feedback;
- `Webcam`: imagem da câmera com landmarks e painel de estado das mãos.

Elementos de feedback implementados:
- Mensagens contextuais (acerto, erro de elemento, bloqueio por escudo, dano recebido);
- Indicador de mira (linha pontilhada com seta);
- Pontuação, recorde e nível;
- Corações para vidas do jogador e dos inimigos.

## 9. Como Executar

### 9.1 Pré-requisitos
- Python 3.10 ou superior;
- Webcam funcional;
- Sistema operacional com suporte ao OpenCV.

### 9.2 Instalação de dependências
```bash
pip install opencv-python mediapipe numpy
```

### 9.3 Execução
Na raiz do projeto:
```bash
python magic_game.py
```

Controles gerais:
- Clique em **INICIAR** na tela inicial;
- Use as mãos para jogar;
- Pressione `ESC` para encerrar.

## 10. Fluxo de Execução (Resumo Técnico)
1. Captura de frame da webcam;
2. Inferência de mãos com MediaPipe;
3. Separação entre mão esquerda e direita;
4. Classificação de mira (esquerda) e gesto (direita);
5. Atualização do estado de jogo (disparos, colisões, ataques inimigos, vidas, pontuação);
6. Renderização das janelas e HUD;
7. Repetição contínua até saída do usuário.

## 11. Decisões de Projeto para Robustez
- Margens de tolerância na leitura dos dedos para reduzir oscilação;
- Lógica de anti-spam no disparo;
- Tratamento de ausência de mão detectada com mensagens de orientação;
- Fallback visual para sprites ausentes (formas geométricas);
- Persistência de recorde com tratamento de erro de leitura/escrita.

## 12. Limitações Atuais
- Dependência de iluminação e enquadramento da câmera;
- Não há calibração por usuário;
- Não há modo multiplayer;
- A interface é feita em janelas OpenCV (não há menu em engine dedicada).

## 13. Possíveis Melhorias Futuras
- Calibração inicial automática de gestos por usuário;
- Detecção temporal com suavização por múltiplos frames;
- Novos elementos, bosses e sistema de fases;
- Coleta de métricas para avaliação experimental (acurácia de gesto, tempo de reação);
- Port para engine de jogos (Unity/Godot) mantendo módulo de visão em Python.

## 14. Conclusão
O projeto valida a viabilidade de usar visão computacional como mecanismo principal de interação em jogos educacionais/experimentais.  
Além do resultado lúdico, a implementação serve como estudo aplicado de reconhecimento de padrões, design de interação e arquitetura modular em Python.