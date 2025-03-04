// Audio Recording and Analysis
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let audioContext;
let analyser;
let scriptProcessor;

// DOM Elements
const recordButton = document.getElementById('recordButton');
const waveform = document.getElementById('waveform');
const result = document.getElementById('result');
const contactForm = document.getElementById('contactForm');

// Enhanced emotion profiles with more features
const emotions = {
    happy: {
        energy: 0.7,
        pitch: 0.8,
        tempo: 0.7,
        variance: 0.6,
        brightness: 0.8,
        rhythmStability: 0.7
    },
    sad: {
        energy: 0.3,
        pitch: 0.3,
        tempo: 0.4,
        variance: 0.3,
        brightness: 0.3,
        rhythmStability: 0.5
    },
    angry: {
        energy: 0.9,
        pitch: 0.8,
        tempo: 0.9,
        variance: 0.8,
        brightness: 0.7,
        rhythmStability: 0.3
    },
    neutral: {
        energy: 0.5,
        pitch: 0.5,
        tempo: 0.5,
        variance: 0.4,
        brightness: 0.5,
        rhythmStability: 0.6
    },
    excited: {
        energy: 0.8,
        pitch: 0.9,
        tempo: 0.8,
        variance: 0.7,
        brightness: 0.9,
        rhythmStability: 0.4
    },
    anxious: {
        energy: 0.6,
        pitch: 0.7,
        tempo: 0.9,
        variance: 0.8,
        brightness: 0.6,
        rhythmStability: 0.3
    }
};

// Advanced audio feature extraction
class AudioFeatureExtractor {
    constructor(audioContext) {
        this.audioContext = audioContext;
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 2048;
        this.bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Float32Array(this.bufferLength);
    }

    extractFeatures(audioData) {
        if (!audioData || audioData.length === 0) {
            console.error('Invalid audio data');
            return null;
        }

        try {
            const features = {
                energy: this.calculateEnergy(audioData),
                pitch: this.estimatePitch(audioData),
                tempo: this.estimateTempo(audioData),
                variance: this.calculateVariance(audioData),
                brightness: this.calculateBrightness(audioData),
                rhythmStability: this.analyzeRhythmStability(audioData)
            };
            return this.normalizeFeatures(features);
        } catch (error) {
            console.error('Error extracting features:', error);
            return null;
        }
    }

    calculateEnergy(audioData) {
        return audioData.reduce((sum, value) => sum + Math.abs(value), 0) / audioData.length;
    }

    estimatePitch(audioData) {
        // Zero-crossing rate method for pitch estimation
        let zeroCrossings = 0;
        for (let i = 1; i < audioData.length; i++) {
            if ((audioData[i] >= 0 && audioData[i - 1] < 0) || 
                (audioData[i] < 0 && audioData[i - 1] >= 0)) {
                zeroCrossings++;
            }
        }
        return (zeroCrossings * this.audioContext.sampleRate) / 
               (2 * audioData.length);
    }

    estimateTempo(audioData) {
        // Onset detection for tempo estimation
        let onsets = 0;
        const threshold = 0.1;
        for (let i = 1; i < audioData.length; i++) {
            if (Math.abs(audioData[i]) > threshold && 
                Math.abs(audioData[i - 1]) <= threshold) {
                onsets++;
            }
        }
        return onsets / (audioData.length / this.audioContext.sampleRate);
    }

    calculateVariance(audioData) {
        const mean = audioData.reduce((sum, value) => sum + value, 0) / audioData.length;
        return audioData.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / audioData.length;
    }

    calculateBrightness(audioData) {
        // Spectral centroid calculation
        this.analyser.getFloatFrequencyData(this.dataArray);
        let sum = 0;
        let weightedSum = 0;
        for (let i = 0; i < this.bufferLength; i++) {
            const magnitude = Math.pow(10, this.dataArray[i] / 20);
            sum += magnitude;
            weightedSum += magnitude * i;
        }
        return weightedSum / sum / this.bufferLength;
    }

    analyzeRhythmStability(audioData) {
        // Analyze rhythm stability through amplitude envelope
        const envelopeLength = 100;
        const envelope = new Float32Array(envelopeLength);
        const samplesPerEnvelope = Math.floor(audioData.length / envelopeLength);
        
        for (let i = 0; i < envelopeLength; i++) {
            const start = i * samplesPerEnvelope;
            const end = start + samplesPerEnvelope;
            let max = 0;
            for (let j = start; j < end; j++) {
                max = Math.max(max, Math.abs(audioData[j]));
            }
            envelope[i] = max;
        }
        
        // Calculate stability through envelope variance
        const envelopeMean = envelope.reduce((sum, value) => sum + value, 0) / envelopeLength;
        const envelopeVariance = envelope.reduce((sum, value) => 
            sum + Math.pow(value - envelopeMean, 2), 0) / envelopeLength;
        
        return 1 - Math.min(envelopeVariance * 10, 1); // Higher stability = lower variance
    }

    normalizeFeatures(features) {
        // Normalize all features to [0,1] range
        const normalized = {};
        for (const [feature, value] of Object.entries(features)) {
            normalized[feature] = Math.min(Math.max(value || 0, 0), 1);
        }
        return normalized;
    }
}

// Emotion classification using weighted features
class EmotionClassifier {
    constructor(emotionProfiles) {
        this.emotionProfiles = emotionProfiles;
        this.weights = {
            energy: 0.25,
            pitch: 0.2,
            tempo: 0.15,
            variance: 0.15,
            brightness: 0.15,
            rhythmStability: 0.1
        };
    }

    classify(features) {
        if (!features) {
            console.error('Invalid features for classification');
            return {
                emotion: 'neutral',
                confidence: 0,
                similarities: {}
            };
        }

        try {
            const similarities = {};
            let maxSimilarity = 0;
            let bestMatch = 'neutral'; // Default emotion

            for (const [emotion, profile] of Object.entries(this.emotionProfiles)) {
                const similarity = this.calculateSimilarity(features, profile);
                similarities[emotion] = similarity;

                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestMatch = emotion;
                }
            }

            return {
                emotion: bestMatch,
                confidence: maxSimilarity,
                similarities
            };
        } catch (error) {
            console.error('Error during classification:', error);
            return {
                emotion: 'neutral',
                confidence: 0,
                similarities: {}
            };
        }
    }

    calculateSimilarity(features, profile) {
        let weightedSimilarity = 0;
        let totalWeight = 0;

        for (const [feature, weight] of Object.entries(this.weights)) {
            if (features[feature] !== undefined && profile[feature] !== undefined) {
                const similarity = 1 - Math.abs(features[feature] - profile[feature]);
                weightedSimilarity += similarity * weight;
                totalWeight += weight;
            }
        }

        return totalWeight > 0 ? weightedSimilarity / totalWeight : 0;
    }
}

// Initialize audio processing
async function initAudioProcessing() {
    try {
        audioContext = new AudioContext();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        
        const featureExtractor = new AudioFeatureExtractor(audioContext);
        const emotionClassifier = new EmotionClassifier(emotions);
        
        return { featureExtractor, emotionClassifier };
    } catch (error) {
        console.error('Error initializing audio processing:', error);
        return null;
    }
}

// Start/Stop Recording
recordButton.addEventListener('click', async () => {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            const audioProcessing = await initAudioProcessing();
            if (!audioProcessing) {
                throw new Error('Failed to initialize audio processing');
            }
            
            const { featureExtractor, emotionClassifier } = audioProcessing;
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                try {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    const audioData = audioBuffer.getChannelData(0);
                    
                    const features = featureExtractor.extractFeatures(audioData);
                    if (!features) {
                        throw new Error('Failed to extract features');
                    }

                    const analysis = emotionClassifier.classify(features);
                    displayResults(analysis, features);
                } catch (error) {
                    console.error('Error processing audio:', error);
                    displayResults(null, null);
                }
            };
            
            mediaRecorder.start();
            isRecording = true;
            recordButton.textContent = 'Stop Recording';
            recordButton.style.background = 'linear-gradient(45deg, #ff0000, #ff5555)';
            
            visualizeAudio(stream);
        } catch (err) {
            console.error('Error accessing microphone:', err);
            alert('Please allow microphone access to use this feature');
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        recordButton.textContent = 'Start Recording';
        recordButton.style.background = 'linear-gradient(45deg, #7928ca, #ff0080)';
    }
});

// Enhanced visualization
function visualizeAudio(stream) {
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    
    source.connect(analyser);
    
    function draw() {
        if (!isRecording) return;
        
        const dataArray = new Float32Array(analyser.frequencyBinCount);
        analyser.getFloatTimeDomainData(dataArray);
        
        const canvas = document.createElement('canvas');
        canvas.width = waveform.clientWidth;
        canvas.height = waveform.clientHeight;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
            console.error('Failed to get canvas context');
            return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#7928ca';
        ctx.beginPath();
        
        const sliceWidth = canvas.width / dataArray.length;
        let x = 0;
        
        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] * 0.5;
            const y = (v * canvas.height / 2) + canvas.height / 2;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }
        
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
        
        waveform.innerHTML = '';
        waveform.appendChild(canvas);
        
        requestAnimationFrame(draw);
    }
    
    draw();
}

// Display analysis results
function displayResults(analysis, features) {
    if (!analysis || !features) {
        result.innerHTML = `
            <div class="bg-white/10 p-4 rounded-lg">
                <p class="text-xl text-red-500">Error analyzing audio</p>
                <p class="text-gray-300">Please try recording again</p>
            </div>
        `;
        return;
    }

    const emotionColors = {
        happy: '#FFD700',
        sad: '#4169E1',
        angry: '#FF4500',
        neutral: '#808080',
        excited: '#FF1493',
        anxious: '#9932CC'
    };

    const emotion = analysis.emotion || 'neutral';
    const color = emotionColors[emotion] || '#7928ca';
    const confidence = analysis.confidence || 0;
    
    result.innerHTML = `
        <h3 class="text-2xl font-bold mb-4">Analysis Result</h3>
        <div class="mb-6">
            <p class="text-xl" style="color: ${color}">
                Primary Emotion: ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}
            </p>
            <p class="text-lg">Confidence: ${Math.round(confidence * 100)}%</p>
        </div>
        
        <div class="grid grid-cols-2 gap-4">
            ${Object.entries(features).map(([feature, value]) => `
                <div class="bg-white/10 p-4 rounded-lg">
                    <div class="text-sm text-gray-300 mb-2">${feature.charAt(0).toUpperCase() + feature.slice(1)}</div>
                    <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div class="h-full transition-all duration-300" 
                             style="width: ${(value || 0) * 100}%; background: ${color}"></div>
                    </div>
                </div>
            `).join('')}
        </div>
        
        <div class="mt-6">
            <h4 class="text-lg mb-3">Other Possible Emotions:</h4>
            <div class="grid grid-cols-2 gap-2">
                ${Object.entries(analysis.similarities || {})
                    .filter(([e]) => e !== emotion && e)
                    .sort(([,a], [,b]) => (b || 0) - (a || 0))
                    .slice(0, 4)
                    .map(([e, similarity]) => `
                        <div class="bg-white/5 p-2 rounded">
                            <span>${e.charAt(0).toUpperCase() + e.slice(1)}</span>
                            <span class="float-right">${Math.round((similarity || 0) * 100)}%</span>
                        </div>
                    `).join('')}
            </div>
        </div>
    `;
}

// Contact Form
contactForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        message: document.getElementById('message').value
    };
    
    try {
        const response = await fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (response.ok) {
            alert('Message sent successfully!');
            contactForm.reset();
        } else {
            throw new Error('Failed to send message');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to send message. Please try again.');
    }
});