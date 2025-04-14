import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {Container, Row, Col, Card, Alert, Spinner, Button, Badge, Form} from 'react-bootstrap';
import axios from 'axios';
import {useProvider} from "../modules/provider";
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
    const [selectedBestExplanation, setSelectedBestExplanation] = useState<ExplanationType | null>(null);
    const { provider, model,providerex,modelex } = useProvider();
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
    const handleSelectBest = async (type: ExplanationType) => {
        try {
            await axios.post(
                'http://localhost:5000/api/track-selection',
                {
                    classificationId,
                    resultId,
                    selectedType: type,
                    timestamp: new Date().toISOString()
                },
                { withCredentials: true }
            );
            setSelectedBestExplanation(type);
        } catch (err) {
            setError('Failed to save selection');
        }
    };
    useEffect(() => {
        setExplanationtext('');
        setPlot('');
        setShapExplanation('');
        setCombinedExplanation('');
        setSelectedBestExplanation(null);
        const fetchData = async () => {
            try {
                // Fetch explanation data
                const explanationResponse = await axios.get(
                    `http://localhost:5000/api/classificationentry/${classificationId}/${resultId}`,
                    { withCredentials: true }
                );
                setClassification(explanationResponse.data);
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
    }, [classificationId, resultId]); // Add resultId to dependency array
    const generateExplanation = async () => {
        setIsExplaining(true);
        // @ts-ignore
        setExplanationtext('');
        try {
            const response = await axios.post('http://localhost:5000/api/explain', {
                truelabel: classification?.actualLabel,
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                text: classification?.text,
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
            // Generate LLM explanation
            setIsExplaining(true);
            const llmResponse = await axios.post('http://localhost:5000/api/explain', {
                truelabel: classification?.actualLabel,
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                text: classification?.text,
                explainer_type: explainerType,
                provider: providerex ,
                model: modelex,

            }, { withCredentials: true } );
            setExplanationtext(llmResponse.data.explanation);

            // Generate SHAP explanation
            const shapResponse = await axios.post('http://localhost:5000/api/explain', {
                truelabel: classification?.actualLabel,
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                text: classification?.text,
                explainer_type: 'shap',


            }, { withCredentials: true } );
            setPlot(shapResponse.data.explanation);
            console.log(shapResponse.data.explanation);
            const shapwords = shapResponse.data.top_words;
            console.log(shapwords,'SHAP words');

            // Generate Combined explanation
            const combinedResponse = await axios.post('http://localhost:5000/api/explain_withshap', {
                text: classification?.text,
                shapwords: shapwords,
                provider: providerex,
                model: modelex,
                label: classification?.prediction,
                confidence: classification?.confidence,
            }, { withCredentials: true } );
            setShapExplanation(combinedResponse.data);
            console.log(combinedResponse.data,'shap llms');

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
                                <div className="mb-4   d-flex flex-column flex-md-row gap-4">
                                    {/* Left: Original Text */}
                                    <div className="flex-grow-1">
                                        <h5 className="mb-3">Original Text</h5>
                                        <p
                                            className="p-3 bg-light rounded mb-2"
                                            style={{
                                                lineHeight: '1.6',
                                                fontSize: '0.95rem',
                                                maxHeight: '300px', // Sabit yükseklik
                                                overflowY: 'auto',  // Scrollbars when needed
                                                whiteSpace: 'pre-wrap', // Satır kaymaları düzgün olsun
                                            }}
                                        >
                                            {classification.text}
                                        </p>
                                        <div style={{ fontSize: '0.75rem', color: '#666' }}>
                                            Confidence: {(classification.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>

                                    {/* Right: Labels */}
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
                    <h5 className="fw-semibold text-muted">Please select the best type of explanation</h5>
                    <hr style={{ width: '60px', margin: '10px auto', borderTop: '2px solid #ccc' }} />
                </Col>
            </Row>
            <div className="position-relative">
                {/* Navigation Arrows */}
                {/* Arrow Container */}
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
                    onClick={() => setActiveExplanation(prev =>
                        prev === 'llm' ? 'combined' : prev === 'shap' ? 'llm' : 'shap'
                    )}
                >
                    <i className="fas fa-chevron-left" />
                </Button>

                {/* Carousel Content */}
                <Row className="g-4 justify-content-center">
                    <Col xs={12} lg={12}>
                        <Card className="h-100">
                            <Card.Body>
                                {/* Explanation Type Selector */}
                                <div className="d-flex justify-content-center gap-2 mb-4">
                                    {['llm', 'shap', 'combined'].map((type) => (
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

                                {/* Explanation Content - now using grid layout! */}
                                <div style={{
                                    minHeight: '300px',
                                    display: 'grid',
                                    gridTemplateColumns: '1fr',
                                    position: 'relative'
                                }}>
                                    {/* LLM */}
                                    <div className={`transition-all ${activeExplanation === 'llm' ? 'opacity-100 z-1' : 'opacity-0 z-0'}`}
                                         style={{
                                             gridArea: '1 / 1 / 2 / 2',
                                             overflowY: 'auto'
                                         }}>
                                        <div className="d-flex justify-content-between align-items-center mb-3">
                                            <Card.Title>LLM Explanation</Card.Title>
                                            <Button
                                                variant="outline-dark"
                                                size="sm"
                                                onClick={() => generateExplanation()}
                                                disabled={isExplaining}
                                            >
                                                {isExplaining ? <Spinner size="sm" /> : 'Generate'}
                                            </Button>
                                        </div>
                                        {explanationtext || 'No explanation generated yet'}
                                        <div className="mt-3 center">
                                            <Button
                                                variant={selectedBestExplanation === 'llm' ? 'success' : 'outline-secondary'}
                                                size="sm"
                                                onClick={() => handleSelectBest('llm')}
                                                disabled={!explanationtext}
                                            >
                                                {selectedBestExplanation === 'llm' ? '✓ Selected as Best' : 'Select as Best'}
                                            </Button>
                                        </div>
                                    </div>

                                    {/* SHAP */}
                                    <div className={`transition-all ${activeExplanation === 'shap' ? 'opacity-100 z-1' : 'opacity-0 z-0'}`}
                                         style={{
                                             gridArea: '1 / 1 / 2 / 2',
                                             overflowY: 'auto'
                                         }}>
                                        <div className="d-flex justify-content-between align-items-center mb-3">
                                            <Card.Title>SHAP Visualization</Card.Title>
                                            <Button
                                                variant="outline-dark"
                                                size="sm"
                                                onClick={() => generateExplanation()}
                                                disabled={isExplaining}
                                            >
                                                {isExplaining ? <Spinner size="sm" /> : 'Generate'}
                                            </Button>
                                        </div>
                                        {plot ? (
                                            <div className="bg-light p-3 rounded">
                                                <div
                                                    dangerouslySetInnerHTML={{ __html: plot }}
                                                    style={{
                                                        maxHeight: '300px',
                                                        overflowY: 'auto'
                                                    }}
                                                />
                                            </div>
                                        ) : (
                                            <div className="text-muted small">
                                                Click generate to view SHAP analysis of important features
                                            </div>
                                        )}
                                        <div className="mt-3">
                                            <Button
                                                variant={selectedBestExplanation === 'shap' ? 'success' : 'outline-secondary'}
                                                size="sm"
                                                onClick={() => handleSelectBest('shap')}
                                                disabled={!plot}
                                            >
                                                {selectedBestExplanation === 'shap' ? '✓ Selected as Best' : 'Select as Best'}
                                            </Button>
                                        </div>
                                    </div>

                                    {/* Combined */}
                                    <div className={`transition-all ${activeExplanation === 'combined' ? 'opacity-100 z-1' : 'opacity-0 z-0'}`}
                                         style={{
                                             gridArea: '1 / 1 / 2 / 2',
                                             overflowY: 'auto'
                                         }}>
                                        <div className="d-flex justify-content-between align-items-center mb-3">
                                            <Card.Title>Combined SHAP + LLM Analysis</Card.Title>
                                            <Button
                                                variant="outline-dark"
                                                size="sm"
                                                onClick={handleGenerateShapExplanation}
                                                disabled={isExplaining || !shapString}
                                            >
                                                {isExplaining ? <Spinner size="sm" /> : 'Generate'}
                                            </Button>
                                        </div>
                                        {shapExplanation ? (
                                            <div className="bg-light p-3 rounded explanation-box">
                  <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', margin: 0 }}>
                    {shapExplanation}
                  </pre>
                                            </div>
                                        ) : (
                                            <div className="text-muted small">
                                                {shapString
                                                    ? "Generate combined explanation using SHAP features and LLM"
                                                    : "Generate SHAP analysis first to enable combined explanation"
                                                }
                                            </div>

                                        )}
                                        <div className="mt-3">
                                            <Button
                                                variant={selectedBestExplanation === 'combined' ? 'success' : 'outline-secondary'}
                                                size="sm"
                                                onClick={() => handleSelectBest('combined')}
                                                disabled={!shapExplanation}
                                            >
                                                {selectedBestExplanation === 'combined' ? '✓ Selected as Best' : 'Select as Best'}
                                            </Button>
                                        </div>
                                    </div>

                                </div>
                            </Card.Body>
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
                        </Card>

                    </Col>
                </Row>
            </div>

            {/* Generate All Button */}
            <div className="text-center mt-4">
                <Button
                    variant="primary"
                    onClick={generateAllExplanations}
                    disabled={isExplaining}
                >
                    {isExplaining ? <Spinner size="sm" className="me-2" /> : null}
                    Generate All Explanations
                </Button>
            </div>

        </Container>
    );
};

export default ExplanationPage;