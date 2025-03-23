import { Button, Form, Card, Alert, Spinner, Badge, ListGroup, ToggleButton, ButtonGroup } from 'react-bootstrap';
import { useAuth } from "../modules/auth";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Classification } from "../types";

const Dashboard = () => {
    const { user, logout } = useAuth();
    const [text, setText] = useState('');
    const [prediction, setPrediction] = useState<{
        id: string;
        label: string;
        score: number;
    } | null>(null);
    const [provider, setProvider] = useState(""); // 'openai' or 'groq'
    const [model, setModel] = useState("");
    const [classifications, setClassifications] = useState<Classification[]>([]);
    const [explanation, setExplanation] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isExplaining, setIsExplaining] = useState(false);
    const [selectedClassification, setSelectedClassification] = useState<string | null>(null);
    const [explainerType, setExplainerType] = useState('llm');
    const groqModels = [
        { name: "allam-2-7b" },
        { name: "deepseek-r1-distill-llama-70b" },
        { name: "deepseek-r1-distill-qwen-32b" },
        { name: "gemma2-9b-it" },
        { name: "llama-3.1-8b-instant" },
        { name: "llama-3.2-11b-vision-preview" },
        { name: "llama-3.2-1b-preview" },
        { name: "llama-3.2-3b-preview" },
        { name: "llama-3.2-90b-vision-preview" },
        { name: "llama-3.3-70b-specdec" },
        { name: "llama-3.3-70b-versatile" },
        { name: "llama-guard-3-8b" },
        { name: "llama3-70b-8192" },
        { name: "llama3-8b-8192" },
        { name: "mistral-saba-24b" },
        { name: "qwen-2.5-32b" },
        { name: "qwen-2.5-coder-32b" },
        { name: "qwen-qwq-32b" }
    ];
    // Function to fetch classifications
    const fetchClassifications = async () => {
        try {
            const response = await fetch('http://localhost:5000/api/classifications', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
            });

            if (response.ok) {
                const data = await response.json();
                if (data.classifications) {
                    setClassifications(data.classifications);
                } else {
                    console.error('No classifications found in the response');
                }
            } else {
                const errorData = await response.json();
                setError(errorData.error || 'Failed to load classifications');
            }
        } catch (err) {
            console.error('Error fetching classifications:', err);
            setError('An error occurred while fetching classifications');
        }
    };

    const analyzeText = async () => {
        setIsLoading(true);
        setError('');
        try {
            const response = await axios.post(
                'http://localhost:5000/api/analyze',
                { text },  // Request body
                { withCredentials: true }  // Include credentials
            );

            setPrediction(response.data);
            setExplanation('');
            setSelectedClassification(null);

            // Refresh classifications after analyzing
            fetchClassifications();
        } catch (err) {
            setError('Failed to analyze text');
            console.error('Error analyzing text:', err);
        }
        setIsLoading(false);
    };

    const generateExplanation = async () => {
        setIsExplaining(true);
        setExplanation('');
        try {
            const response = await axios.post('http://localhost:5000/api/explain', {
                prediction_id: prediction?.id,
                text: text,
                explainer_type: explainerType,
                provider: provider ,
                model: model,

            }, { withCredentials: true } );

            if (explainerType === 'shap') {
                setExplanation(response.data.explanation);
            } else {
                setExplanation(response.data.explanation); // Normal metin geldiğinde kaydet.
            }
        } catch (err) {
            setError('Failed to generate explanation');
        }
        setIsExplaining(false);
    };

    // View a previous classification (without explanation)
    const viewPreviousClassification = (classification: Classification) => {
        setSelectedClassification(classification.id);
        setPrediction({
            id: classification.id,
            label: classification.label,
            score: classification.score
        });
        setText(classification.text);
        setExplanation(''); // Clear any existing explanation
    };

    // Initial fetch of classifications
    useEffect(() => {
        fetchClassifications();
    }, []);

    // Format the timestamp in a readable way
    const formatDate = (timestamp: string | number | Date) => {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        return date.toLocaleString();
    };

    // Get sentiment badge color
    const getSentimentBadge = (label: string, score: number) => {
        if (label === 'POSITIVE') {
            return <Badge bg="success">Positive ({(score).toFixed(1)}%)</Badge>;
        } else {
            return <Badge bg="danger">Negative ({(score).toFixed(1)}%)</Badge>;
        }
    };

    // Truncate long text for display
    const truncateText = (str: string, maxLength = 80) => {
        return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
    };

    return (
        <div className="py-5">
            <div className="hero-section mb-5 text-center">
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

            <div className="container">
                <div className="row">
                    <div className="col-lg-7">
                        <Card className="shadow mb-4">
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
                                        <Card className="mb-3 border-primary">
                                            <Card.Body>
                                                <h5>Analysis Result</h5>
                                                <p className="mb-0">
                                                    Sentiment: {prediction.label === 'POSITIVE' ? (
                                                    <span className="text-success">Positive</span>
                                                ) : (
                                                    <span className="text-danger">Negative</span>
                                                )}
                                                </p>
                                                <p>Confidence: {(prediction.score).toFixed(1)}%</p>

                                                <div className="d-flex align-items-center mb-3">
                                                    <span className="me-3">Explainer Type:</span>
                                                    <ButtonGroup>
                                                        <ToggleButton
                                                            id="explainer-llm"
                                                            type="radio"
                                                            variant={explainerType === 'llm' ? 'primary' : 'outline-primary'}
                                                            name="explainer"
                                                            value="llm"
                                                            checked={explainerType === 'llm'}
                                                            onChange={(e) => setExplainerType(e.currentTarget.value)}
                                                        >
                                                            LLM
                                                        </ToggleButton>
                                                        <ToggleButton
                                                            id="explainer-shap"
                                                            type="radio"
                                                            variant={explainerType === 'shap' ? 'primary' : 'outline-primary'}
                                                            name="explainer"
                                                            value="shap"
                                                            checked={explainerType === 'shap'}
                                                            onChange={(e) => setExplainerType(e.currentTarget.value)}
                                                        >
                                                            SHAP
                                                        </ToggleButton>
                                                    </ButtonGroup>
                                                </div>

                                                {/* Show provider options if LLM is selected */}
                                                {explainerType === 'llm' && (
                                                    <div className="mb-3">
                                                        <span className="me-3">Select Provider:</span>
                                                        <ButtonGroup>
                                                            <ToggleButton
                                                                id="provider-openai"
                                                                type="radio"
                                                                variant={provider === 'openai' ? 'primary' : 'outline-primary'}
                                                                name="provider"
                                                                value="openai"
                                                                checked={provider === 'openai'}
                                                                onChange={(e) => setProvider(e.currentTarget.value)}
                                                            >
                                                                OpenAI
                                                            </ToggleButton>
                                                            <ToggleButton
                                                                id="provider-groq"
                                                                type="radio"
                                                                variant={provider === 'groq' ? 'primary' : 'outline-primary'}
                                                                name="provider"
                                                                value="groq"
                                                                checked={provider === 'groq'}
                                                                onChange={(e) => setProvider(e.currentTarget.value)}
                                                            >
                                                                Groq
                                                            </ToggleButton>
                                                        </ButtonGroup>
                                                    </div>
                                                )}

                                                {/* Show model selection only if Groq is chosen */}
                                                {provider === 'groq' && (
                                                    <div className="mb-3">
                                                        <span className="me-3">Select Model:</span>
                                                        <Form.Select value={model} onChange={(e) => setModel(e.target.value)}>
                                                            <option value="">-- Select a Model --</option>
                                                            {groqModels.map((m) => (
                                                                <option key={m.name} value={m.name}>
                                                                    {m.name}
                                                                </option>
                                                            ))}
                                                        </Form.Select>
                                                    </div>
                                                )}

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
                                                            <span
                                                                className="ms-2">Generating {explainerType.toUpperCase()} Explanation...</span>
                                                        </>
                                                    ) : `Explain with ${explainerType.toUpperCase()}`}
                                                </Button>
                                            </Card.Body>
                                        </Card>
                                    </div>
                                )}

                                {explanation && (
                                    <Card className="mt-3 border-info">
                                        <Card.Body>
                                            <Card.Title>
                                                {explainerType === 'shap' ? 'SHAP Explanation' : 'Explanation'}
                                            </Card.Title>
                                            <div className="text-muted">
                                                {explainerType === 'shap' ? (
                                                    // SHAP HTML görselleştirmesi
                                                    <div
                                                        dangerouslySetInnerHTML={{__html: explanation}}
                                                        style={{
                                                            fontFamily: 'monospace',
                                                            fontSize: '14px',
                                                            lineHeight: '1.5',
                                                            overflowX: 'auto',
                                                            minHeight: '100px'  // Boş görünmesin diye minimum yükseklik ekle
                                                        }}
                                                        className="shap-html-container"
                                                    />
                                                ) : (
                                                    // LLM text explanation
                                                    <p style={{whiteSpace: 'pre-wrap'}}>{explanation}</p>
                                                )}
                                            </div>
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

                    <div className="col-lg-5">
                        <Card className="shadow">
                            <Card.Header className="bg-primary text-white">
                                <h4 className="mb-0">Previous Classifications</h4>
                            </Card.Header>
                            <Card.Body className="p-0" style={{maxHeight: "630px", overflowY: "auto"}}>
                                {classifications.length > 0 ? (
                                    <ListGroup variant="flush">
                                        {classifications.map(classification => (
                                            <ListGroup.Item
                                                key={classification.id}
                                                className={`border-bottom py-3 ${selectedClassification === classification.id ? 'bg-light' : ''}`}
                                            >
                                                <div className="d-flex justify-content-between align-items-center mb-2">
                                                    {getSentimentBadge(classification.label, classification.score)}
                                                    <small
                                                        className="text-muted">{formatDate(classification.timestamp)}</small>
                                                </div>
                                                <p className="mb-2" title={classification.text}>
                                                    {truncateText(classification.text)}
                                                </p>
                                                <Button
                                                    variant="outline-secondary"
                                                    size="sm"
                                                    onClick={() => viewPreviousClassification(classification)}
                                                    className="me-2"
                                                >
                                                    View
                                                </Button>
                                            </ListGroup.Item>
                                        ))}
                                    </ListGroup>
                                ) : (
                                    <div className="p-4 text-center text-muted">
                                        <p>No previous classifications found</p>
                                    </div>
                                )}
                            </Card.Body>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;