import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {Container, Row, Col, Card, Alert, Spinner, Button, Badge, Form} from 'react-bootstrap';
import axios from 'axios';
import {useProvider} from "../modules/provider";

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
    const { provider, model,providerex,modelex } = useProvider();
    const { datasetId, classificationId, resultId } = useParams();
    const [explanationtext, setExplanationtext] = useState('');
    const [explanation, setExplanation] = useState<ExplanationData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isExplaining, setIsExplaining] = useState(false);
    const [explainerType, setExplainerType] = useState<'llm' | 'shap'>('llm');
    const [plot, setPlot] = useState('');
    const [shapString, setShapString] = useState('');
    const [shapExplanation, setShapExplanation] = useState('');
    const navigate = useNavigate();
    const [totalResults, setTotalResults] = useState(0);
    const [currentResultIndex, setCurrentResultIndex] = useState(0);
    const [shapstring, setShapstring] = useState('');

    useEffect(() => {
        setExplanationtext('');
        setPlot('');
        const fetchData = async () => {
            console.log(providerex,modelex,'models and stuff')
            try {
                // Fetch explanation data
                const explanationResponse = await axios.get(
                    `http://localhost:5000/api/explanation/${classificationId}/${resultId}`,
                    { withCredentials: true }
                );
                setExplanation(explanationResponse.data);
                console.log(explanationResponse.data);

                // Fetch classification metadata to get total results
                const classificationResponse = await axios.get(
                    `http://localhost:5000/api/classification/${classificationId}`,
                    { withCredentials: true }
                );

                // Use optional chaining and default to empty array if results is undefined
                const resultsLength = classificationResponse.data.results?.length || 0;
                setTotalResults(resultsLength);

                // Ensure resultId is parsed correctly, default to 0 if invalid
                const parsedResultId = Number(resultId) || 0;
                setCurrentResultIndex(parsedResultId);

            } catch (err) {
                setError("Failed to load data");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [classificationId, resultId,currentResultIndex]); // Add resultId to dependency array
    const generateExplanation = async () => {
        setIsExplaining(true);
        // @ts-ignore
        setExplanationtext('');
        try {
            const response = await axios.post('http://localhost:5000/api/explain', {
                truelabel: explanation?.actualLabel,
                predictedlabel: explanation?.prediction,
                confidence: explanation?.confidence,
                text: explanation?.text,
                explainer_type: explainerType,
                provider: providerex ,
                model: modelex,

            }, { withCredentials: true } );

            if (explainerType === 'shap') {
                setPlot(response.data.explanation);
                setShapstring(response.data.top_words);
            } else {
                setExplanationtext(response.data.explanation); // Normal metin geldiğinde kaydet.
            }
        } catch (err) {
            setError('Failed to generate explanation');
            console.log(err);
        }
        setIsExplaining(false);
    };
    const handlePrevious = () => {
        const newIndex = currentResultIndex - 1;
        navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
    };

    const handleNext = () => {
        const newIndex = currentResultIndex + 1;
        navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
    };

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
                setPlot(response.data.explanation);
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
            <div className="d-flex justify-content-between mb-4">
                <Button
                    variant="outline-secondary"
                    onClick={() => navigate(`/datasets/${datasetId}/classifications/${classificationId}`)}
                >
                    ← Back to Classification
                </Button>

                <div className="d-flex gap-2">
                    <Button
                        variant="outline-primary"
                        onClick={handlePrevious}
                        disabled={currentResultIndex === 0}
                    >
                        ← Previous
                    </Button>

                    <Button
                        variant="outline-primary"
                        onClick={handleNext}
                        disabled={currentResultIndex >= totalResults - 1}
                    >
                        Next →
                    </Button>
                </div>
            </div>

            <div className="text-center mb-4 text-muted">
                Result {currentResultIndex + 1} of {totalResults}
            </div>


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
                                <div className="mb-4   d-flex flex-column flex-md-row gap-4">
                                    {/* Left: Original Text */}
                                    <div className="flex-grow-1">
                                        <h5 className="mb-3">Original Text</h5>
                                        <p className="p-3 bg-light rounded mb-2" style={{ lineHeight: '1.6', fontSize: '0.95rem' }}>
                                            {explanation.text}
                                        </p>
                                        <div style={{ fontSize: '0.75rem', color: '#666' }}>
                                            Confidence: {(explanation.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>

                                    {/* Right: Labels */}
                                    <div className="d-flex flex-column justify-content-center align-items-center">
                                        <div className="mb-3 text-center">
                                            <div className="text-muted mb-1" style={{ fontSize: '0.85rem' }}>Prediction</div>
                                            <div
                                                className={`d-flex justify-content-center align-items-center px-4 py-2 rounded-pill fw-semibold text-white shadow-sm ${explanation.prediction === 'POSITIVE' ? 'bg-success' : 'bg-danger'}`}
                                                style={{ fontSize: '0.9rem', minWidth: '120px', height: '38px' }}
                                            >
                                                {explanation.prediction}
                                            </div>
                                        </div>

                                        <div className="text-center">
                                            <div className="text-muted mb-1" style={{ fontSize: '0.85rem' }}>Actual Label</div>
                                            <div
                                                className={`d-flex justify-content-center align-items-center px-4 py-2 rounded-pill fw-semibold text-white shadow-sm ${explanation.actualLabel === 1 ? 'bg-success' : 'bg-danger'}`}
                                                style={{ fontSize: '0.9rem', minWidth: '120px', height: '38px' }}
                                            >
                                                {explanation.actualLabel === 1 ? 'POSITIVE' : 'NEGATIVE'}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                            </Card.Body>
                        </Card>
                    </Col>
                </Row>


            ) : null}
            <Row className="mb-4">
                <Col md={8} className="mx-auto">
                </Col>
            </Row>
            <div className="bg-white rounded-4 shadow-sm p-4 mb-4 border border-light-subtle">
                <div className="mb-3 d-flex flex-column flex-md-row align-items-start align-items-md-center justify-content-between">
                    <h5 className="mb-2 mb-md-0">Explainer Type</h5>
                    <div className="d-flex gap-2">
                        <Button
                            variant={explainerType === 'llm' ? 'dark' : 'outline-dark'}
                            onClick={() => setExplainerType('llm')}
                        >
                            LLM
                        </Button>
                        {explanation?.provider===null && (
                            <Button
                                variant={explainerType === 'shap' ? 'dark' : 'outline-dark'}
                                onClick={() => setExplainerType('shap')}
                            >
                                SHAP
                            </Button>
                        )}
                    </div>
                </div>

                <hr className="my-3" />

                <div className="text-center">
                    <Button
                        variant="dark"
                        size="lg"
                        className="px-4"
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
                                <span className="ms-2">
                        Generating {explainerType.toUpperCase()} Explanation...
                    </span>
                            </>
                        ) : (
                            `Explain with ${explainerType.toUpperCase()}`
                        )}
                    </Button>
                </div>
            </div>
            {(explanationtext!=='' || plot) && (
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
                                <p style={{whiteSpace: 'pre-wrap'}}>{explanationtext}</p>
                            )}
                        </div>
                    </Card.Body>
                </Card>
            )}

        </Container>
    );
};

export default ExplanationPage;