import { Box, Button, Paper, Typography } from '@mui/material';
import { useState } from 'react';
import axios from 'axios';
import './DevActions.style.css';

export default function DevActions(props: { messages: string[] }) {
    const [isRetraining, setIsRetraining] = useState<boolean>(false);

    const handleRetrainModels = async (mode: 'both' | 'detector' | 'recognizer') => {
        try {
            setIsRetraining(true);
            await axios.post(`http://localhost:8000/api/retrain/${mode}`);
        } catch (error) {
            alert("Error during retraining. Check backend console.");
        } finally {
            setIsRetraining(false);
        }
    }

    return (
        <Box id="dev-actions">
            <Typography id="dev-actions-title">
                Dev Actions
            </Typography>
            <Box id="dev-actions-content">
                <Box id="dev-actions-buttons">
                    <Button 
                        variant="outlined"
                        className='retrain-button' 
                        onClick={() => handleRetrainModels('both')} 
                        disabled={isRetraining}
                    >
                        Retrain Both Models
                    </Button>
                    <Button 
                        variant="outlined"
                        className='retrain-button' 
                        onClick={() => handleRetrainModels('detector')} 
                        disabled={isRetraining}
                    >
                        Retrain Detection Model
                    </Button>
                    <Button 
                        variant="outlined"
                        className='retrain-button' 
                        onClick={() => handleRetrainModels('recognizer')} 
                        disabled={isRetraining}
                    >
                        Retrain Recognition Model
                    </Button>
                </Box>
                <Box id="dev-actions-logs">
                    <Paper id="dev-actions-logs-paper" elevation={3}>
                        <Typography id="dev-actions-logs-title">
                            Shell logs
                        </Typography>
                        <Box>
                            {props.messages.map((msg, index) => (
                                <Typography key={index}>
                                    {msg}
                                </Typography>
                            ))}
                        </Box>
                    </Paper>
                </Box>
            </Box>
        </Box>
    );
}