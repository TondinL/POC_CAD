import random
import math

# ----------------------
# Compose (come prima)
# ----------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for i, t in enumerate(self.transforms):
            if isinstance(x, list) and x and isinstance(x[0], list):
                nxt = Compose(self.transforms[i:])
                return [nxt(clip) for clip in x]
            x = t(x)
        return x


# ----------------------
# CROP deterministici (senza padding)
# ----------------------
class TemporalBeginCropStrict:
    """Primi N frame. Se non ci sono N consecutivi, solleva errore."""
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        n = len(frame_indices)
        if n < self.size:
            raise ValueError(f"Sequenza troppo corta: {n} < {self.size}")
        return frame_indices[:self.size]


class TemporalCenterCropStrict:
    """N frame centrati. Nessun padding: fallisce se troppo corta."""
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        n = len(frame_indices)
        if n < self.size:
            raise ValueError(f"Sequenza troppo corta: {n} < {self.size}")
        center = n // 2
        begin = max(0, min(center - self.size // 2, n - self.size))
        end = begin + self.size
        return frame_indices[begin:end]


# ----------------------
# Random crop con retry (senza padding)
# ----------------------
class TemporalRandomCropStrict:
    """
    Sceglie casualmente un inizio valido in [0, n-size].
    Se non trova (o n<size), fallisce. Niente padding.
    """
    def __init__(self, size, max_retries=10):
        self.size = size
        self.max_retries = max_retries

    def __call__(self, frame_indices):
        n = len(frame_indices)
        if n < self.size:
            raise ValueError(f"Sequenza troppo corta: {n} < {self.size}")

        max_begin = n - self.size
        # range valido deterministico: se max_retries == 0, prendo directly un begin casuale valido
        for _ in range(max(1, self.max_retries)):
            begin = random.randint(0, max_begin)
            end = begin + self.size
            clip = frame_indices[begin:end]
            if len(clip) == self.size:
                return clip

        # Come fallback, prendo un begin deterministico valido (es. center-based)
        begin = max(0, (n - self.size) // 2)
        end = begin + self.size
        clip = frame_indices[begin:end]
        if len(clip) == self.size:
            return clip

        # Se ancora non va (caso patologico), errore esplicito:
        raise RuntimeError("Impossibile trovare un crop valido senza padding.")


# ----------------------
# Multi-clip “strict”: nessun padding, scarta clip corte
# ----------------------
class TemporalEvenCropStrict:
    """
    Crea n_samples clip di lunghezza size, distribuite uniformemente entro [0, n-size].
    Nessun padding: se n<size, errore. Se n_samples>possibile, riduce.
    """
    def __init__(self, size, n_samples=1):
        assert n_samples >= 1
        self.size = size
        self.n_samples = n_samples

    def __call__(self, frame_indices):
        n = len(frame_indices)
        if n < self.size:
            raise ValueError(f"Sequenza troppo corta: {n} < {self.size}")

        max_begin = n - self.size
        if self.n_samples == 1:
            begin = max(0, (n - self.size) // 2)
            return [frame_indices[begin:begin + self.size]]

        # calcolo begin uniformi in [0, max_begin]
        if self.n_samples == 2:
            begins = [0, max_begin]
        else:
            step = max_begin / (self.n_samples - 1) if self.n_samples > 1 else 0
            begins = [int(round(i * step)) for i in range(self.n_samples)]

        out = []
        for b in begins:
            clip = frame_indices[b:b + self.size]
            if len(clip) == self.size:
                out.append(clip)
        # se qualche clip non entra, ci accontentiamo di quelle valide
        if not out:
            raise RuntimeError("EvenCropStrict: nessuna clip valida.")
        return out


class SlidingWindowStrict:
    """
    Finestre scorrevoli di lunghezza size, passo stride.
    Nessun padding: scarta l’ultima se corta. stride=0 -> stride=size.
    """
    def __init__(self, size, stride=0):
        self.size = size
        self.stride = size if stride == 0 else stride

    def __call__(self, frame_indices):
        n = len(frame_indices)
        if n < self.size:
            raise ValueError(f"Sequenza troppo corta: {n} < {self.size}")

        out = []
        for start in range(0, n - self.size + 1, self.stride):
            clip = frame_indices[start:start + self.size]
            if len(clip) == self.size:
                out.append(clip)
        if not out:
            raise RuntimeError("SlidingWindowStrict: nessuna finestra valida.")
        return out


# ----------------------
# (Opzionale) Subsampling: attenzione, altera la velocità apparente
# ----------------------
class TemporalSubsampling:
    def __init__(self, stride):
        assert stride >= 1
        self.stride = stride
    def __call__(self, frame_indices):
        return frame_indices[::self.stride]