import { Button, Form, Card, Alert, Spinner } from 'react-bootstrap';
import {useAuth} from "../modules/auth";
import React, { useState } from 'react';
import axios from 'axios';


const Dashboard = () => {
    const { user, logout } = useAuth();
    const [text, setText] = useState('');
    const [prediction, setPrediction] = useState<{
        id: string;
        label: string;
        score: number;
    } | null>(null);
    const [explanation, setExplanation] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isExplaining, setIsExplaining] = useState(false);
    const analyzeText = async () => {
        setIsLoading(true);
        setError('');
        try {
            const response = await axios.post('http://localhost:5000/api/analyze', {
                text
            });

            setPrediction(response.data);
            setExplanation('');
        } catch (err) {
            setError('Failed to analyze text');
        }
        setIsLoading(false);
    };
    const generateExplanation = async () => {
        setIsExplaining(true);
        try {
            const response = await axios.post('http://localhost:5000/api/explain', {
                prediction_id: prediction?.id,
                text
            });

            setExplanation(response.data.explanation);
        } catch (err) {
            setError('Failed to generate explanation');
        }
        setIsExplaining(false);
    };
    return (
        <div className="py-5 text-center">
            <div className="hero-section mb-5">

                {user ? (
                    <h1 className="display-4 mb-3">
                        Welcome {user?.username || 'to Auth App'}
                    </h1>
                ) : (
                    <div className="mt-4">
                        <a href="/login" className="btn btn-primary mx-2">
                            Login
                        </a>
                        <a href="/register" className="btn btn-outline-primary mx-2">
                            Register
                        </a>
                    </div>
                )}
            </div>
            <Card className="shadow mx-auto mt-5" style={{ maxWidth: '800px' }}>
                <Card.Body>
                    <Card.Title className="mb-4">Sentiment Analyzer</Card.Title>

                    <Form.Group>
                        <Form.Control
                            as="textarea"
                            rows={3}
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Enter your text here..."
                            className="mb-3"
                        />
                    </Form.Group>

                    <div className="d-grid gap-2">
                        <Button
                            variant="primary"
                            onClick={analyzeText}
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <Spinner
                                        as="span"
                                        animation="border"
                                        size="sm"
                                        role="status"
                                        aria-hidden="true"
                                    />
                                    <span className="ms-2">Analyzing...</span>
                                </>
                            ) : 'Analyze Sentiment'}
                        </Button>
                    </div>

                    {prediction && (
                        <div className="mt-4">
                            <Card className="mb-3">
                                <Card.Body>
                                    <h5>Analysis Result</h5>
                                    <p className="mb-0">
                                        Sentiment: {prediction.label === 'POSITIVE' ? (
                                        <span className="text-success">Positive</span>
                                    ) : (
                                        <span className="text-danger">Negative</span>
                                    )}
                                    </p>
                                    <p>Confidence: {(prediction.score * 100).toFixed(1)}%</p>

                                    <Button
                                        variant="info"
                                        onClick={generateExplanation}
                                        disabled={isExplaining}
                                    >
                                        {isExplaining ? (
                                            <>
                                                <Spinner
                                                    as="span"
                                                    animation="border"
                                                    size="sm"
                                                    role="status"
                                                    aria-hidden="true"
                                                />
                                                <span className="ms-2">Generating...</span>
                                            </>
                                        ) : 'Explain Result'}
                                    </Button>
                                </Card.Body>
                            </Card>
                        </div>
                    )}

                    {explanation && (
                        <Card className="mt-3 border-info">
                            <Card.Body>
                                <Card.Title>Explanation</Card.Title>
                                <p className="text-muted">{explanation}</p>
                            </Card.Body>
                        </Card>
                    )}

                    {error && (
                        <Alert variant="danger" className="mt-3">
                            {error}
                        </Alert>
                    )}
                </Card.Body>
            </Card>
        </div>


    );
};
export default Dashboard;