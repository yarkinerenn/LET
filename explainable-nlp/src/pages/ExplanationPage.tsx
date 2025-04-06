import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Alert, Spinner, Button, Badge } from 'react-bootstrap';
import axios from 'axios';

interface ExplanationData {
    text: string;
    prediction: string;
    confidence: number;
    actualLabel?: number;
    explanation: string;
    importantWords?: { word: string; score: number }[];
    provider?: string;
    model?: string;
}

const ExplanationPage = () => {
    const { datasetId, classificationId, resultId } = useParams();
    const [explanation, setExplanation] = useState<ExplanationData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isExplaining, setIsExplaining] = useState(false);
    const [explainerType, setExplainerType] = useState<'llm' | 'shap'>('llm');
    const [shapPlot, setShapPlot] = useState('');
    const [shapString, setShapString] = useState('');
    const [shapExplanation, setShapExplanation] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        const fetchExplanation = async () => {
            try {
                const response = await axios.get(
                    `http://localhost:5000/api/explanation/${classificationId}/${resultId}`,
                    { withCredentials: true }
                );
                setExplanation(response.data);
            } catch (err) {
                setError("Failed to load explanation");
            } finally {
                setLoading(false);
            }
        };

        fetchExplanation();
    }, [classificationId, resultId]);

    const handleGenerateExplanation = async () => {
        if (!explanation) return;

        setIsExplaining(true);
        try {
            const response = await axios.post(
                'http://localhost:5000/api/explain',
                {
                    text: explanation.text,
                    prediction_id: resultId,
                    explainer_type: explainerType,
                    provider: explanation.provider,
                    model: explanation.model
                },
                { withCredentials: true }
            );

            if (explainerType === 'shap') {
                setShapPlot(response.data.explanation);
                setShapString(response.data.top_words);
            } else {
                setExplanation(prev => prev ? {
                    ...prev,
                    explanation: response.data.explanation
                } : null);
            }
        } catch (err) {
            setError('Failed to generate explanation');
        }
        setIsExplaining(false);
    };


    const handleGenerateShapExplanation = async () => {
        setIsExplaining(true);
        try {
            const response = await axios.post(
                'http://localhost:5000/api/explain_withshap',
                {
                    text: explanation?.text,
                    shapwords: shapString,
                    provider: explanation?.provider,
                    model: explanation?.model
                },
                { withCredentials: true }
            );
            setShapExplanation(response.data);
        } catch (err) {
            setError('Failed to generate SHAP explanation');
        }
        setIsExplaining(false);
    };

    return (
        <Container className="py-4">
            <Button
                variant="outline-secondary"
                onClick={() => navigate(`/datasets/${datasetId}/classifications/${classificationId}`)}
                className="mb-4"
            >
                ← Back to Classification
            </Button>

            {loading ? (
                <div className="text-center py-5">
                    <Spinner animation="border" />
                </div>
            ) : error ? (
                <Alert variant="danger">{error}</Alert>
            ) : explanation ? (
                <Row>
                    <Col md={8} className="mx-auto">
                        <Card>
                            <Card.Body>

                                <div className="mb-4">
                                    <h5>Original Text</h5>
                                    <p className="p-3 bg-light rounded">{explanation.text}</p>
                                </div>

                                <Row className="mb-4">
                                    <Col md={6}>
                                        <div className="border rounded p-3 text-center bg-white shadow-sm">
                                            <div className="text-muted mb-1" style={{ fontSize: '0.85rem' }}>Prediction</div>
                                            <div
                                                className={`d-inline-block px-3 py-1 rounded-pill fw-semibold text-white ${explanation.prediction === 'POSITIVE' ? 'bg-success' : 'bg-danger'}`}
                                                style={{ fontSize: '0.8rem' }}
                                            >
                                                {explanation.prediction}
                                            </div>
                                            <div className="mt-2" style={{ fontSize: '0.75rem', color: '#666' }}>
                                                Confidence: {(explanation.confidence * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    </Col>

                                    {explanation.actualLabel !== undefined && (
                                        <Col md={6}>
                                            <div className="border rounded p-3 text-center bg-white shadow-sm">
                                                <div className="text-muted mb-1" style={{ fontSize: '0.85rem' }}>Actual Label</div>
                                                <div
                                                    className={`d-inline-block px-3 py-1 rounded-pill fw-semibold text-white ${explanation.actualLabel === 1 ? 'bg-success' : 'bg-danger'}`}
                                                    style={{ fontSize: '0.8rem' }}
                                                >
                                                    {explanation.actualLabel === 1 ? 'POSITIVE' : 'NEGATIVE'}
                                                </div>
                                            </div>
                                        </Col>
                                    )}
                                </Row>

                                <div className="mb-4">
                                    <h5>Explanation Controls</h5>
                                    <div className="d-flex gap-2 mb-3">
                                        <Button
                                            variant={explainerType === 'llm' ? 'dark' : 'outline-dark'}
                                            onClick={() => setExplainerType('llm')}
                                        >
                                            LLM Explanation
                                        </Button>
                                        <Button
                                            variant={explainerType === 'shap' ? 'dark' : 'outline-dark'}
                                            onClick={() => setExplainerType('shap')}
                                        >
                                            SHAP Visualization
                                        </Button>
                                    </div>

                                    <Button
                                        variant="dark"
                                        onClick={handleGenerateExplanation}
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
                                                <span className="ms-2">
                                                    Generating {explainerType.toUpperCase()} Explanation...
                                                </span>
                                            </>
                                        ) : `Generate ${explainerType.toUpperCase()} Explanation`}
                                    </Button>
                                </div>

                                {explainerType === 'llm' && explanation.explanation && (
                                    <Card className="mb-4">
                                        <Card.Body>
                                            <Card.Title>AI Explanation</Card.Title>
                                            <p className="lead">{explanation.explanation}</p>
                                        </Card.Body>
                                    </Card>
                                )}

                                {explainerType === 'shap' && shapPlot && (
                                    <>
                                        <Card className="mb-4">
                                            <Card.Body>
                                                <Card.Title>SHAP Visualization</Card.Title>
                                                <div
                                                    dangerouslySetInnerHTML={{ __html: shapPlot }}
                                                    style={{
                                                        fontFamily: 'monospace',
                                                        fontSize: '14px',
                                                        lineHeight: '1.5',
                                                        overflowX: 'auto'
                                                    }}
                                                />
                                            </Card.Body>
                                        </Card>

                                        <Button
                                            variant="secondary"
                                            onClick={handleGenerateShapExplanation}
                                            disabled={isExplaining}
                                            className="mb-3"
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
                                                    <span className="ms-2">Generating Enhanced Explanation...</span>
                                                </>
                                            ) : "Explain SHAP Visualization with LLM"}
                                        </Button>

                                        {shapExplanation && (
                                            <Card>
                                                <Card.Body>
                                                    <Card.Title>Enhanced SHAP Explanation</Card.Title>
                                                    <p style={{ whiteSpace: 'pre-wrap' }}>{shapExplanation}</p>
                                                </Card.Body>
                                            </Card>
                                        )}
                                    </>
                                )}

                                {explanation.importantWords && (
                                    <Card>
                                        <Card.Body>
                                            <Card.Title>Key Influencing Words</Card.Title>
                                            <div className="d-flex flex-wrap gap-2">
                                                {explanation.importantWords.map((word, index) => (
                                                    <Badge
                                                        key={index}
                                                        bg={word.score > 0 ? 'success' : 'danger'}
                                                        className="p-2"
                                                    >
                                                        {word.word} ({word.score > 0 ? '+' : ''}{word.score.toFixed(2)})
                                                    </Badge>
                                                ))}
                                            </div>
                                        </Card.Body>
                                    </Card>
                                )}
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
            ) : null}
        </Container>
    );
};

export default ExplanationPage;