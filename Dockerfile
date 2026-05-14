FROM oven/bun:latest AS builder
WORKDIR /app

COPY package.json .
COPY ./src/files/references.json.gz .
COPY ./src/convert.ts .

RUN gunzip references.json.gz && bun run convert.ts

FROM oven/bun:slim
WORKDIR /app

COPY --from=builder /app/vectors.bin .
COPY --from=builder /app/labels.bin .
COPY --from=builder /app/vptree.bin .
COPY ./src/index.ts .
COPY ./src/vector.ts .
COPY ./src/types.ts .
COPY ./src/files/mcc_risk.json ./files/mcc_risk.json
COPY ./src/files/normalization.json ./files/normalization.json

LABEL maintainer="João Marcos <joaomarcostomaz70@gmail.com>"
LABEL description="API de Risco de Fraude para a Rinha de Backend 2026 em Bun"

EXPOSE 3000

CMD ["bun", "run", "index.ts"]