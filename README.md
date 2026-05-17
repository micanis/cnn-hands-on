# CNN Hands-on

CNN (Convolutional Neural Network) ハンズオン教材

## Structure

```
apps/
  web/      # Astro frontend
  api/      # Go backend
  slides/   # Slidev presentation
workshop/   # Python notebooks & exercises
```

## Setup

```bash
# devbox + direnv
direnv allow

# or manually
devbox shell
```

## Development

```bash
devbox run dev:web       # Astro frontend (localhost:4321)
devbox run dev:api       # Go backend
devbox run dev:slides    # Slidev presentation
devbox run workshop:setup     # Python environment setup
devbox run workshop:jupyter   # Start Jupyter
```
