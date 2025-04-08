import {Button, Form, Card, Alert, Spinner, Badge, ListGroup, ToggleButton, ButtonGroup, Table} from 'react-bootstrap';
import { useAuth } from "../modules/auth";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Classification } from "../types";
import {useNavigate} from "react-router-dom";
import {useProvider} from "../modules/provider";

const Dashboard = () => {
    const { provider, model,providerex,modelex } = useProvider();
    const { user } = useAuth();
    const [text, setText] = useState('');
    const [prediction, setPrediction] = useState<{
        id: string;
        label: string;
        score: number;
    } | null>(null);
    const navigate = useNavigate(); // Define useNavigate correctly
    useEffect(() => {
        console.log(user)
        if (!user) {
            navigate('/login');
        }
    }, [user, navigate]);
    const [classifications, setClassifications] = useState<Classification[]>([]);
    const [explanation, setExplanation] = useState('');
    const [shap_explanation, setShap_Explanation] = useState('');
    const [plot, setPlot] = useState('');
    const [shapstring, setShapstring] = useState('');

    const [datasets, setDatasets] = useState<any[]>([]);

    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isExplaining, setIsExplaining] = useState(false);
    const [selectedClassification, setSelectedClassification] = useState<string | null>(null);
    const [explainerType, setExplainerType] = useState('llm');

    const fetchDatasets = async () => {
        try {
            const response = await axios.get("http://localhost:5000/api/datasets", {
                withCredentials: true,
            });
            setDatasets(response.data.datasets);
        } catch (err) {
            setError("Failed to load datasets.");
        }
    };
    const handleViewDataset = (datasetId: string) => {
        navigate(`/dataset/${datasetId}`);
    };
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
    const analyzeTextwithLLM = async () => {
        setIsLoading(true);
        setError('');
        try {
            const response = await axios.post(
                'http://localhost:5000/api/analyzewithllm',
                { text:text,
                    provider: provider ,
                    model: model,},  // Request body
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
    const generateExplanationfromshap = async () => {
        setIsExplaining(true);
        setExplanation('');
        try {
            console.log('shapstr',shapstring);
            const response = await axios.post('http://localhost:5000/api/explain_withshap', {
                prediction_id: prediction?.id,
                text: text,
                explainer_type: explainerType,
                provider: providerex ,
                model: modelex,
                shapwords: shapstring

            }, { withCredentials: true } );
            setShap_Explanation(response.data);


        } catch (err) {
            setError('Failed to generate explanation');
        }
        setIsExplaining(false);
    };

    const generateExplanation = async () => {
        setIsExplaining(true);
        setExplanation('');
        try {
            const response = await axios.post('http://localhost:5000/api/explain', {
                prediction_id: prediction?.id,
                text: text,
                explainer_type: explainerType,
                provider: providerex ,
                model: modelex,

            }, { withCredentials: true } );

            if (explainerType === 'shap') {
                setPlot(response.data.explanation);
                setShapstring(response.data.top_words);
            } else {
                setExplanation(response.data.explanation); // Normal metin geldiğinde kaydet.
            }
        } catch (err) {
            setError('Failed to generate explanation');
        }
        setIsExplaining(false);
    };
    const handleDeleteClassification = async (classificationId: string) => {
        try {
            // Send delete request to the backend
            const response = await axios.delete(`http://localhost:5000/api/delete_prediction/${classificationId}`, {
                withCredentials: true
            });

            if (response.status === 200) {
                // If deletion is successful, filter out the deleted classification from the state
                setClassifications((prevClassifications) =>
                    prevClassifications.filter((classification) => classification.id !== classificationId)
                );
            } else {
            }
        } catch (err) {
            console.error("Error deleting classification:", err);
            alert("An error occurred while deleting the classification.");
        }
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
        fetchDatasets();

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
            return <Badge bg="success">Positive ({(score).toFixed(3)})</Badge>;
        } else {
            return <Badge bg="danger">Negative ({(score).toFixed(3)})</Badge>;
        }
    };

    // Truncate long text for display
    const truncateText = (str: string, maxLength = 80) => {
        return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
    };

    return (
        <div className="py-5">

            <div className="container">
                <div className="row">
                    <div className="col-lg-7">
                        <h3 className="text-center mb-3">My Datasets</h3>
                        <Table striped bordered hover>
                            <thead>
                            <tr>
                                <th>#</th>
                                <th>Filename</th>
                                <th>Uploaded At</th>
                                <th>Actions</th>
                            </tr>
                            </thead>
                            <tbody>
                            {datasets.map((dataset, index) => (
                                <tr key={dataset._id}>
                                    <td>{index + 1}</td>
                                    <td>
                                        <Button variant="link" onClick={() => handleViewDataset(dataset._id)}>
                                            {dataset.filename}
                                        </Button>
                                    </td>
                                    <td>{new Date(dataset.uploaded_at).toLocaleString()}</td>
                                    <td>
                                        <Button
                                            variant="danger"
                                            size="sm"
                                            onClick={async () => {
                                                await axios.delete(`http://localhost:5000/api/delete_dataset/${dataset._id}`, {
                                                    withCredentials: true,
                                                });
                                                fetchDatasets();
                                            }}
                                        >
                                            Delete
                                        </Button>
                                    </td>
                                </tr>
                            ))}
                            </tbody>
                        </Table>
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

                                <div className="d-grid gap-2 d-md-flex justify-content-md-start">
                                    {/* Sentiment Analyze Button */}
                                    <Button
                                        variant="dark"
                                        onClick={analyzeText}
                                        disabled={isLoading}
                                        className="me-md-2 mb-2 mb-md-0"
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
                                        ) : (
                                            'Analyze with BERT'
                                        )}
                                    </Button>

                                    {/* LLM Analyze Button */}
                                    <Button
                                        variant="dark"
                                        onClick={analyzeTextwithLLM}
                                        disabled={isLoading}
                                        className="mb-2 mb-md-0"
                                    >
                                        Analyze with LLM
                                    </Button>
                                </div>

                                {prediction && (
                                    <div className="mt-4">
                                        <Card className="mb-3 border-dark-subtle">
                                            <Card.Body>
                                                <h5>Analysis Result</h5>
                                                <p className="mb-0">
                                                    Sentiment: {prediction.label === 'POSITIVE' ? (
                                                    <span className="text-success">Positive</span>
                                                ) : (
                                                    <span className="text-danger">Negative</span>
                                                )}
                                                </p>
                                                <p>Confidence: {(prediction.score)}</p>



                                                <div className="d-flex align-items-center mb-3">
                                                    <span className="me-3">Explainer Type:</span>

                                                    {/* LLM Button */}
                                                    <Button
                                                        variant={explainerType === 'llm' ? 'dark' : 'outline-dark'}
                                                        onClick={() => setExplainerType('llm')}
                                                        className="me-2"
                                                    >
                                                        LLM
                                                    </Button>

                                                    {/* SHAP Button */}
                                                    <Button
                                                        variant={explainerType === 'shap' ? 'dark' : 'outline-dark'}
                                                        onClick={() => setExplainerType('shap')}
                                                    >
                                                        SHAP
                                                    </Button>
                                                </div>




                                                <Button
                                                    variant="dark"
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


                            </Card.Body>
                        </Card>
                    </div>
                    <div className="col-lg-5">
                        <Card className="shadow">
                            <Card.Header className="bg-dark
                             text-white">
                                <h4 className="mb-0">Previous Classifications</h4>
                            </Card.Header>
                            <Card.Body className="p-0" style={{ maxHeight: "600px", overflowY: "auto" }}>
                                {classifications.length > 0 ? (
                                    <ListGroup variant="flush">
                                        {classifications.map((classification) => (
                                            <ListGroup.Item
                                                key={classification.id}
                                                className={`border-bottom py-3 ${selectedClassification === classification.id ? 'bg-light' : ''}`}
                                            >
                                                <div className="d-flex justify-content-between align-items-center mb-2">
                                                    {getSentimentBadge(classification.label, classification.score)}
                                                    <small className="text-muted">{formatDate(classification.timestamp)}</small>
                                                </div>
                                                <p className="mb-2" title={classification.text}>
                                                    {truncateText(classification.text)}
                                                </p>
                                                <div className="d-flex justify-content-between">
                                                    <Button
                                                        variant="outline-secondary"
                                                        size="sm"
                                                        onClick={() => viewPreviousClassification(classification)}
                                                        className="me-2"
                                                    >
                                                        View
                                                    </Button>
                                                    {/* Delete Button */}
                                                    <Button
                                                        variant="outline-danger"
                                                        size="sm"
                                                        onClick={() => handleDeleteClassification(classification.id)}  // Call delete function
                                                    >
                                                        Delete
                                                    </Button>
                                                </div>
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
                    {(explanation!=='' || plot) && (
                        <Card className="mt-3 border-dark-subtle">
                            <Card.Body>
                                <Card.Title>
                                    {explainerType === 'shap' ? 'SHAP Explanation' : 'Explanation'}
                                </Card.Title>
                                <div className="text-muted">
                                    {explainerType === 'shap' ? (
                                        // SHAP HTML görselleştirmesi

                                        <div
                                            dangerouslySetInnerHTML={{__html: plot}}
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
                    {/* Show "Explain SHAP with LLM" button only if SHAP explanation exists */}
                    {plot && explainerType === 'shap' &&(
                        <Button
                            variant="secondary"
                            className="mt-3"
                            onClick={generateExplanationfromshap}
                            disabled={isExplaining}
                        >
                            {isExplaining ? (
                                <>
                                    <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true"/>
                                    <span className="ms-2">Explaining SHAP with LLM...</span>
                                </>
                            ) : "Explain SHAP with LLM"}
                        </Button>
                    )}
                    {shap_explanation && explainerType === 'shap' && (
                        <div className="mt-3 p-3 border rounded bg-light">
                            <h6>SHAP LLM Explanation</h6>
                            <p style={{whiteSpace: "pre-wrap"}}>{shap_explanation}</p>
                        </div>
                    )}

                    {error && (
                        <Alert variant="danger" className="mt-3">
                            {error}
                        </Alert>
                    )}


                </div>
            </div>
        </div>
    );
};

export default Dashboard;