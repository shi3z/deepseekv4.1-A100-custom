// Web Audio API Synthesizer for Jev Arcade 20 (Zero external dependencies, iOS Safari compatible)

class SoundFX {
    constructor() {
        this.ctx = null;
        this.enabled = true;
    }

    init() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                this.ctx = new AudioCtx();
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    playTone(freq, duration, type = 'sine', gainVal = 0.15) {
        if (!this.enabled) return;
        this.init();
        if (!this.ctx) return;

        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
            gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);

            osc.connect(gain);
            gain.connect(this.ctx.destination);

            osc.start();
            osc.stop(this.ctx.currentTime + duration);
        } catch (e) {
            console.warn("Audio playback error:", e);
        }
    }

    click() {
        this.playTone(800, 0.05, 'square', 0.08);
    }

    select() {
        this.playTone(520, 0.08, 'triangle', 0.12);
        setTimeout(() => this.playTone(780, 0.1, 'triangle', 0.12), 60);
    }

    startPlay() {
        this.playTone(440, 0.08, 'sine', 0.1);
        setTimeout(() => this.playTone(554.37, 0.08, 'sine', 0.1), 70);
        setTimeout(() => this.playTone(659.25, 0.15, 'sine', 0.12), 140);
    }

    success() {
        this.playTone(523.25, 0.1, 'triangle', 0.15); // C5
        setTimeout(() => this.playTone(659.25, 0.1, 'triangle', 0.15), 100); // E5
        setTimeout(() => this.playTone(783.99, 0.1, 'triangle', 0.15), 200); // G5
        setTimeout(() => this.playTone(1046.50, 0.25, 'triangle', 0.18), 300); // C6
    }

    critical() {
        this.playTone(880, 0.08, 'sawtooth', 0.15);
        setTimeout(() => this.playTone(1108.73, 0.08, 'sawtooth', 0.15), 80);
        setTimeout(() => this.playTone(1318.51, 0.2, 'sawtooth', 0.18), 160);
    }

    failure() {
        this.playTone(300, 0.15, 'sawtooth', 0.15);
        setTimeout(() => this.playTone(220, 0.25, 'sawtooth', 0.15), 150);
    }

    toggle() {
        this.enabled = !this.enabled;
        return this.enabled;
    }
}

window.soundFX = new SoundFX();
