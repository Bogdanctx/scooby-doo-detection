import { useEffect, useRef, useState } from 'react'
import { 
    Box, 
    Typography, 
} from '@mui/material'
import './App.css'
import type { DetectionResult } from './interfaces'
import DevActions from './DevActions'
import ResultsSection from './ResultsSection'
import UploadSection from './UploadSection'

function App() {
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
    const [messages, setMessages] = useState<string[]>([]);
    
    const websocket = useRef<WebSocket | null>(null);

    useEffect(() => {
        websocket.current = new WebSocket('ws://localhost:8000/ws');

        websocket.current.onopen = () => {
            console.log('WebSocket connection established');
        };

        websocket.current.onmessage = (event) => {
            console.log('WebSocket message received:', event.data);
            setMessages((prevMessages) => [...prevMessages, event.data]);
        };

        websocket.current.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        websocket.current.onclose = () => {
            console.log('WebSocket connection closed');
        };

        return () => {
            websocket.current?.close();
        };
    }, []);

	return (
		<>
			{/* Header */}
			<Box id="header" sx={{ width: '100%', textAlign: 'center', padding: '1rem' }}>
				<Typography variant="h2" align="center" sx={{ fontSize: '2rem', color: 'rgb(15, 118, 110)' }}>
					Scooby-Doo Detection
				</Typography>
				<Typography variant="h5" sx={{ fontSize: '0.8rem', color: 'rgb(71, 85, 105)' }}>
					Upload an image to identify the Mystery Inc. gang!
				</Typography>
			</Box>

			{/* Main Content */}
			<Box id="main-content" sx={{ width: '100%', pb: 8 }}>
                <UploadSection setDetectionResult={setDetectionResult} />
                <ResultsSection detectionResult={detectionResult} />
                <DevActions messages={messages} />
			</Box>

			{/* Footer */}
			<Box id="footer">
				
			</Box>
		</>
	)
}

export default App