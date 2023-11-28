# node base image
FROM node:lts-alpine AS builder

# Set the working directory
WORKDIR /app

# Copy your application files to the container
COPY NeoML .

# Install Langium
RUN npm i -g yo generator-langium

# Generate Langium generator
RUN npm install
RUN npm run langium:generate
RUN npm run build

# ------------------ #

# Base image
FROM node:lts AS base

# Set the working directory
WORKDIR /app

# Update
RUN apt-get update

# Install R and required libraries
RUN apt-get install -y r-base
RUN R -e "install.packages(c('e1071', 'class', 'rpart', 'nnet', 'caret', 'dplyr', 'tidyr'), repos='http://cran.rstudio.com/')"

# Install Python 
RUN apt-get install -y python3-full python3-pip
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip3 install pandas scikit-learn

# Copy the application bin from builder
COPY --from=builder /app/bin ./NeoML/bin
COPY --from=builder /app/out ./NeoML/out
COPY --from=builder /app/src/test ./NeoML/src/test
COPY --from=builder /app/node_modules ./NeoML/node_modules
COPY --from=builder /app/package.json ./NeoML
COPY --from=builder /app/vite.config.ts ./NeoML

# Copy the application tests
COPY  fuzzer ./fuzzer
COPY  datasets ./datasets
COPY Programs_examples/ ./Programs_examples

# Set the entry point command
CMD ["/bin/bash"]
