import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Alert, Spinner, Button, Badge, Form } from 'react-bootstrap';
import axios from 'axios';
import { useProvider } from "../modules/provider";
import '../index.css';

interface ClassificationEntry {
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
    const { provider, model, providerex, modelex } = useProvider();
    const { datasetId, classificationId, resultId } = useParams();
    const [explanationtext, setExplanationtext] = useState('');
    const [classification, setClassification] = useState<ClassificationEntry | null>(null);
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
    const [combinedExplanation, setCombinedExplanation] = useState('');
    type ExplanationType = 'llm' | 'shap' | 'combined';
    const [activeExplanation, setActiveExplanation] = useState<ExplanationType>('llm');
    const [ratings, setRatings] = useState({
        llm: 0,
        shap: 0,
        combined: 0
    });
    const [isSubmittingRatings, setIsSubmittingRatings] = useState(false);

    const handleRatingChange = (type: ExplanationType, rating: number) => {
        setRatings(prev => ({
            ...prev,
            [type]: rating
        }));
    };

    const submitRatings = async () => {
        setIsSubmittingRatings(true);
        try {
            await axios.post(
                'http://localhost:5000/api/save_ratings',
                {
                    classificationId,
                    resultId,
                    ratings,
                    timestamp: new Date().toISOString()
                },
                { withCredentials: true }
            );
            // Optionally show success message or navigate away
        } catch (err) {
            setError('Failed to submit ratings');
        } finally {
            setIsSubmittingRatings(false);
        }
    };

    useEffect(() => {
        setExplanationtext('');
        setPlot('');
        setShapExplanation('');
        setCombinedExplanation('');
        setRatings({ llm: 0, shap: 0, combined: 0 }); // Reset ratings when changing result
        const fetchData = async () => {
            try {
                const explanationResponse = await axios.get(
                    `http://localhost:5000/api/classificationentry/${classificationId}/${resultId}`,
                    { withCredentials: true }
                );
                setClassification(explanationResponse.data);
                setPlot(explanationResponse.data.shap_plot);
                setExplanationtext(explanationResponse.data.llm_explanation);
                setShapExplanation(explanationResponse.data.shapwithllm);
                const existingRatings = explanationResponse.data.ratings;
                if (existingRatings) {
                    setRatings({
                        llm: existingRatings.llm || 0,
                        shap: existingRatings.shap || 0,
                        combined: existingRatings.combined || 0
                    });
                }

                const classificationResponse = await axios.get(
                    `http://localhost:5000/api/classification/${classificationId}`,
                    { withCredentials: true }
                );

                const resultsLength = classificationResponse.data.results?.length || 0;
                setTotalResults(resultsLength);
                const parsedResultId = Number(resultId) || 0;
                setCurrentResultIndex(parsedResultId);

            } catch (err) {
                setError("Failed to load data");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [classificationId, resultId]);

    const generateExplanation = async () => {
        setIsExplaining(true);
        try {
            const response = await axios.post('http://localhost:5000/api/explain', {
                truelabel: classification?.actualLabel,
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                text: classification?.text,
                explainer_type: explainerType,
                provider: providerex,
                model: modelex,
            }, { withCredentials: true });

            if (explainerType === 'shap') {
                setPlot(response.data.explanation);
                setShapstring(response.data.top_words);
            } else {
                setExplanationtext(response.data.explanation);
            }
        } catch (err) {
            setError('Failed to generate explanation');
            console.log(err);
        }
        setIsExplaining(false);
    };

    const handlePrevious = () => {
        const newIndex = currentResultIndex - 1;
        setCurrentResultIndex(newIndex);
        navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
    };

    const handleNext = () => {
        const newIndex = currentResultIndex + 1;
        setCurrentResultIndex(newIndex);
        navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
    };

    const generateAllExplanations = async () => {
        try {
            setIsExplaining(true);
            const llmResponse = await axios.post('http://localhost:5000/api/explain', {
                truelabel: classification?.actualLabel,
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                text: classification?.text,
                explainer_type: 'llm',
                classificationId: classificationId,
                resultId: resultId,
            }, { withCredentials: true });
            setExplanationtext(llmResponse.data.explanation);

            const shapResponse = await axios.post('http://localhost:5000/api/explain', {
                truelabel: classification?.actualLabel,
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                text: classification?.text,
                explainer_type: 'shap',
                classificationId: classificationId,
                resultId: resultId
            }, { withCredentials: true });
            setPlot(shapResponse.data.explanation);
            const shapwords = shapResponse.data.top_words;

            const combinedResponse = await axios.post('http://localhost:5000/api/explain_withshap', {
                text: classification?.text,
                shapwords: shapwords,
                provider: providerex,
                model: modelex,
                label: classification?.prediction,
                confidence: classification?.confidence,
                classificationId: classificationId,
                resultId: resultId,
            }, { withCredentials: true });
            setShapExplanation(combinedResponse.data);
        } catch (err) {
            setError('Failed to generate explanations');
        } finally {
            setIsExplaining(false);
        }
    };

    const handleGenerateShapExplanation = async () => {
        setIsExplaining(true);
        try {
            const response = await axios.post(
                'http://localhost:5000/api/explain_withshap',
                {
                    text: classification?.text,
                    shapwords: shapString,
                    provider: providerex,
                    model: modelex,
                    label: classification?.prediction,
                    confidence: classification?.confidence,
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
            ) : classification ? (
                <Row>
                    <Col md={8} className="mx-auto">
                        <Card>
                            <Card.Body>
                                <div className="mb-4 d-flex flex-column flex-md-row gap-4">
                                    <div className="flex-grow-1">
                                        <h5 className="mb-3">Original Text</h5>
                                        <p
                                            className="p-3 bg-light rounded mb-2"
                                            style={{
                                                lineHeight: '1.6',
                                                fontSize: '0.95rem',
                                                maxHeight: '300px',
                                                overflowY: 'auto',
                                                whiteSpace: 'pre-wrap',
                                            }}
                                        >
                                            {classification.text}
                                        </p>
                                        <div style={{ fontSize: '0.75rem', color: '#666' }}>
                                            Confidence: {(classification.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>

                                    <div className="d-flex flex-column justify-content-center align-items-center">
                                        <div className="mb-3 text-center">
                                            <div className="text-muted mb-1" style={{ fontSize: '0.85rem' }}>Prediction</div>
                                            <div
                                                className={`d-flex justify-content-center align-items-center px-4 py-2 rounded-pill fw-semibold text-white shadow-sm ${classification.prediction === 'POSITIVE' ? 'bg-success' : 'bg-danger'}`}
                                                style={{ fontSize: '0.9rem', minWidth: '120px', height: '38px' }}
                                            >
                                                {classification.prediction}
                                            </div>
                                        </div>

                                        <div className="text-center">
                                            <div className="text-muted mb-1" style={{ fontSize: '0.85rem' }}>Actual Label</div>
                                            <div
                                                className={`d-flex justify-content-center align-items-center px-4 py-2 rounded-pill fw-semibold text-white shadow-sm ${classification.actualLabel === 1 ? 'bg-success' : 'bg-danger'}`}
                                                style={{ fontSize: '0.9rem', minWidth: '120px', height: '38px' }}
                                            >
                                                {classification.actualLabel === 1 ? 'POSITIVE' : 'NEGATIVE'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </Card.Body>
                        </Card>
                    </Col>
                </Row>
            ) : null}

            <Row className="mb-5 mt-4 text-center">
                <Col md={12}>
                    <h5 className="fw-semibold text-muted">Please rate each explanation (1-5)</h5>
                    <hr style={{ width: '60px', margin: '10px auto', borderTop: '2px solid #ccc' }} />
                </Col>
            </Row>

            <div className="position-relative">
                {classification?.model !== 'llm' && (
                    <Button
                        variant="light"
                        className="position-absolute d-flex align-items-center justify-content-center"
                        style={{
                            left: '-40px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            zIndex: 10,
                            width: '36px',
                            height: '36px',
                            borderRadius: '50%',
                            boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
                            padding: 0
                        }}
                        onClick={() =>
                            setActiveExplanation(prev =>
                                prev === 'llm' ? 'combined' : prev === 'shap' ? 'llm' : 'shap'
                            )
                        }
                    >
                        <i className="fas fa-chevron-left" />
                    </Button>
                )}

                <Row className="g-4 justify-content-center">
                    <Col xs={12} lg={12}>
                        <Card className="h-100">
                            <Card.Body>
                                <div className="d-flex justify-content-center gap-2 mb-4">
                                    {['llm', 'shap', 'combined']
                                        .filter((type) => {
                                            if (classification?.model === 'llm') {
                                                return type === 'llm';
                                            }
                                            return true;
                                        })
                                        .map((type) => (
                                            <Button
                                                key={type}
                                                variant={activeExplanation === type ? 'dark' : 'outline-dark'}
                                                size="sm"
                                                onClick={() => setActiveExplanation(type as ExplanationType)}
                                            >
                                                {type.toUpperCase()}
                                            </Button>
                                        ))}
                                </div>

                                <div style={{
                                    minHeight: '300px',
                                    display: 'grid',
                                    gridTemplateColumns: '1fr',
                                    position: 'relative'
                                }}>
                                    {['llm', 'shap', 'combined']
                                        .filter(type => classification?.model === 'llm' ? type === 'llm' : true)
                                        .map((type) => (
                                            <div
                                                key={type}
                                                className={`transition-all ${activeExplanation === type ? 'opacity-100 z-1' : 'opacity-0 z-0'}`}
                                                style={{
                                                    gridArea: '1 / 1 / 2 / 2',
                                                    overflowY: 'auto'
                                                }}
                                            >
                                                <div className="d-flex justify-content-between align-items-center mb-3">
                                                    <Card.Title>
                                                        {type === 'llm' ? 'LLM Explanation' :
                                                            type === 'shap' ? 'SHAP Visualization' :
                                                                'Combined SHAP + LLM Analysis'}
                                                    </Card.Title>
                                                    <Button
                                                        variant="outline-dark"
                                                        size="sm"
                                                        onClick={() => {
                                                            if (type === 'combined') {
                                                                handleGenerateShapExplanation();
                                                            } else {
                                                                setExplainerType(type as 'llm' | 'shap');
                                                                generateExplanation();
                                                            }
                                                        }}
                                                        disabled={isExplaining}
                                                    >
                                                        {isExplaining ? <Spinner size="sm" /> : 'Generate'}
                                                    </Button>
                                                </div>

                                                {type === 'llm' && (
                                                    <div className="mb-3">
                                                        {explanationtext || 'No explanation generated yet'}
                                                    </div>
                                                )}

                                                {type === 'shap' && (plot ? (
                                                    <div className="bg-light p-3 rounded mb-3">
                                                        <div
                                                            dangerouslySetInnerHTML={{ __html: plot }}
                                                            style={{
                                                                maxHeight: '300px',
                                                                overflowY: 'auto'
                                                            }}
                                                        />
                                                    </div>
                                                ) : (
                                                    <div className="text-muted small mb-3">
                                                        Click generate to view SHAP analysis of important features
                                                    </div>
                                                ))}

                                                {type === 'combined' && (shapExplanation ? (
                                                    <div className="bg-light p-3 rounded explanation-box mb-3">
                                                        <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', margin: 0 }}>
                                                            {shapExplanation}
                                                        </pre>
                                                    </div>
                                                ) : (
                                                    <div className="text-muted small mb-3">
                                                        {shapString
                                                            ? "Generate combined explanation using SHAP features and LLM"
                                                            : "Generate SHAP analysis first to enable combined explanation"
                                                        }
                                                    </div>
                                                ))}

                                                <div className="mt-3">
                                                    <h6>Rate this explanation (1-5):</h6>
                                                    <div className="d-flex justify-content-center gap-1">
                                                        {[1, 2, 3, 4, 5].map((star) => (
                                                            <Button
                                                                key={star}
                                                                variant={ratings[type as ExplanationType] >= star ? "warning" : "outline-secondary"}
                                                                className="p-1"
                                                                style={{ minWidth: '36px' }}
                                                                onClick={() => handleRatingChange(type as ExplanationType, star)}
                                                                disabled={!(
                                                                    (type === 'llm' && explanationtext) ||
                                                                    (type === 'shap' && plot) ||
                                                                    (type === 'combined' && shapExplanation))
                                                                }
                                                            >
                                                                {star}
                                                            </Button>
                                                        ))}
                                                    </div>
                                                    <div className="text-center mt-2 small text-muted">
                                                        {ratings[type as ExplanationType] ? `You rated: ${ratings[type as ExplanationType]}` : 'Not rated yet'}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </Card.Body>

                            {classification?.model !== 'llm' && (
                                <Button
                                    variant="light"
                                    className="position-absolute d-flex align-items-center justify-content-center"
                                    style={{
                                        right: '-40px',
                                        top: '50%',
                                        transform: 'translateY(-50%)',
                                        zIndex: 10,
                                        width: '36px',
                                        height: '36px',
                                        borderRadius: '50%',
                                        boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
                                        padding: 0
                                    }}
                                    onClick={() => setActiveExplanation(prev =>
                                        prev === 'llm' ? 'shap' : prev === 'shap' ? 'combined' : 'llm'
                                    )}
                                >
                                    <i className="fas fa-chevron-right" />
                                </Button>
                            )}
                        </Card>
                    </Col>
                </Row>
            </div>

            <div className="text-center mt-4">
                <Button
                    variant="primary"
                    onClick={generateAllExplanations}
                    disabled={isExplaining}
                    className="me-2"
                >
                    {isExplaining ? <Spinner size="sm" className="me-2" /> : null}
                    Generate All Explanations
                </Button>

                <Button
                    variant="success"
                    onClick={submitRatings}
                    disabled={isSubmittingRatings || Object.values(ratings).every(rating => rating === 0)}
                >
                    {isSubmittingRatings ? (
                        <Spinner size="sm" className="me-2" />
                    ) : null}
                    Submit All Ratings
                </Button>
            </div>
        </Container>
    );
};

export default ExplanationPage;