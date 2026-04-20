# Jogo: Mago dos Elementos (Controle por Mãos)

## Conceito Geral

O jogador controla um mago em um duelo contra inimigos elementais utilizando duas mãos com visão computacional:

- Mão esquerda: controla a direção da mira (como se fosse a varinha)
- Mão direita: define a ação (tipo de feitiço ou defesa)

A proposta é substituir o conceito de "dois analógicos" (como em twin-stick shooters) pelo uso das duas mãos.

---

## Mecânica Principal

### Controle

- A posição da mão esquerda define a direção dos ataques
- A mão direita define a ação com base em gestos

### Ações (mão direita)

Exemplo de mapeamento de gestos:

- Punho fechado → projétil de fogo
- Dois dedos → projétil de água ou gelo
- Um dedo → raio
- Mão aberta (segurando) → escudo

Os gestos podem ser ajustados conforme a facilidade de detecção.

---

## Sistema de Combate

### Projéteis

Cada ação gera um projétil com:
- tipo elemental
- direção baseada na mão esquerda
- velocidade e comportamento próprios

### Inimigos

Tipos de inimigos:
- Fogo
- Gelo
- Elétrico

Cada inimigo possui:
- tipo elemental
- posição fixa no topo da tela
- comportamento simples (atacar após um tempo)

### Fraquezas Elementais

Exemplo de lógica:
- Fogo é derrotado por gelo/água
- Gelo é derrotado por fogo
- Elétrico é derrotado por terra ou outro elemento escolhido

O jogador precisa escolher o feitiço correto para derrotar cada inimigo.

---

## Sistema de Defesa

- Mão direita aberta ativa um escudo
- O escudo bloqueia projéteis inimigos
- Sem escudo, o jogador perde vida ao ser atingido

---

## Estrutura de Jogo

### Layout

- Jogador fixo na parte inferior da tela
- Inimigos posicionados no topo em slots fixos
- Área central livre para projéteis

### Comportamento dos Inimigos

- Inimigos aparecem em posições fixas
- Podem:
  - esperar um tempo (carregar ataque)
  - lançar projéteis contra o jogador
- Não se movimentam livremente para evitar confusão visual

---

## Progressão

Sistema baseado em ondas (waves):

- Wave 1:
  - 1 tipo de inimigo
  - baixa velocidade

- Wave 2:
  - 2 tipos de inimigos
  - mais frequência de spawn

- Wave 3+:
  - múltiplos tipos
  - maior velocidade
  - maior quantidade de inimigos

A progressão aumenta a dificuldade sem necessidade de mudar o cenário.

---

## Feedback ao Jogador

Elementos simples:

- Indicador de direção (linha ou seta)
- Texto de feedback:
  - "Acerto correto"
  - "Tipo errado"
  - "Defesa"
- Pontuação
- Vida do jogador
- Número da wave

---

## Assets Necessários

- Background (cenário)
- Player (mago)
- Inimigos:
  - fogo
  - gelo
  - elétrico
- Projéteis:
  - fogo
  - gelo
  - elétrico
- Escudo
- Indicador de direção (opcional)

---

## Objetivo

Sobreviver o maior número de ondas possível derrotando inimigos com o feitiço correto, utilizando coordenação entre direção (mão esquerda) e ação (mão direita).