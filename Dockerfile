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
RUN R -e "install.packages(c('e1071', 'class', 'rpart', 'nnet', 'caret'), repos='http://cran.rstudio.com/')"

# Install Python 
RUN apt-get install -y python3-full python3-pip
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip3 install pandas scikit-learn

# Copy the application bin from builder
COPY --from=builder /app/bin ./bin
COPY --from=builder /app/out ./out
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json .
COPY --from=builder /app/tests ./tests
COPY --from=builder /app/tests.sh .

# Install the dependencies
RUN npm install

# Set the entry point command
CMD ["/bin/bash"]
