# Sistema de Controle por Dedos (Refinado)

## Conceito

Separar em duas etapas:

1. Seleção do poder (mostrar qual elemento está ativo)
2. Execução do poder (quando lançar)

Isso evita que o jogador tenha que decorar combinações complexas.

---

## Mapeamento de Gestos

### Seleção de Poder (mão direita)

- Indicador levantado → seleciona FOGO
- Dedo polegar levantado → seleciona ÁGUA
- Mindinho levantado → seleciona ELÉTRICO

Observação:
- Apenas um dedo ativo por vez (simplifica muito)
- Quando o dedo está levantado, o poder está "preparado"

---

### Execução do Poder

Opções recomendadas:

#### Opção A (mais simples)
- Abaixar o dedo → dispara o poder

Fluxo:
1. Levanta o dedo → seleciona
2. Abaixa → dispara

---

#### Opção B (mais controlada)
- Levantar dedo → seleciona
- Fazer um gesto de "empurrar" com a mão → dispara

(mais difícil de implementar, pode deixar pra depois)

---

### Defesa

- Mão aberta → ativa escudo
- Escudo fica ativo enquanto a mão estiver aberta

---

## Feedback Visual (ESSENCIAL)

O jogador precisa saber qual poder está selecionado.

Sugestões simples:

- Cor ao redor da mão:
  - Vermelho → fogo
  - Azul → gelo
  - Amarelo → elétrico

- Texto na tela:
  - "FIRE READY"
  - "ICE READY"
  - "LIGHTNING READY"

---

## Lógica de Estado

Você vai precisar de um estado atual:

- selected_element
- last_finger_state

Exemplo:

selected_element = FIRE
is_finger_up = true

Quando:
is_finger_up muda de true → false

→ dispara projétil

---

## Importante (para evitar bugs)

### 1. Debounce (anti-spam)

Sem isso, vai disparar várias vezes.

Solução:
- só dispara quando detecta transição:
  dedo levantado → dedo abaixado

---

### 2. Tolerância de detecção

Às vezes o MediaPipe oscila.

Solução:
- considerar o dedo "levantado" só se estiver consistente por alguns frames

---

### 3. Prioridade de gestos

Se mais de um dedo aparecer levantado:

- ignorar OU
- escolher um (ex: prioridade: indicador > médio > mindinho)

---

## Loop de Interação

1. Detecta mão direita
2. Verifica qual dedo está levantado
3. Atualiza selected_element
4. Detecta mudança (levantado → abaixado)
5. Dispara projétil na direção da mão esquerda

---

## Vantagens desse sistema

- Não precisa decorar combinações
- Feedback visual claro
- Separação mental simples:
  - escolher → executar
- Muito mais confiável com visão computacional