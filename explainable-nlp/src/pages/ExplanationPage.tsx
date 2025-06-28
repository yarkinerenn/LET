import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Alert, Spinner, Button, Tab, Tabs, Badge } from 'react-bootstrap';
import axios from 'axios';
import { useProvider } from "../modules/provider";
import '../index.css';

interface ClassificationEntry {
    text: string;
    prediction: string;
    confidence: number;
    actualLabel?: number;
    explanation: string;
    llm_explanations?: Record<string, { content: string }>;
    shap_plot_explanation?: { content: string };
    shapwithllm_explanations?: Record<string, { content: string }>;
    importantWords?: { word: string; score: number }[];
    provider?: string;
    model?: string;
    explanation_models?: Array<{provider: string, model: string}>; // Add this line
}

interface ExplanationData {
    llm?: string;
    combined?: string;
}

interface ShapData {
    explanation?: string;
    shapWords?: string[];
}

interface ModelInfo {
    model: string;
    id: string;
    name: string;
    provider: string;
}

const ExplanationPage = () => {
    const [faithfulnessScore, setFaithfulnessScore] = useState<number | null>(null);
    const [isFetchingFaithfulness, setIsFetchingFaithfulness] = useState(false);
    const [faithfulnessError, setFaithfulnessError] = useState<string | null>(null);
    const { datasetId, classificationId, resultId } = useParams();
    const navigate = useNavigate();
    const [classification, setClassification] = useState<ClassificationEntry | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isExplaining, setIsExplaining] = useState(false);
    const [totalResults, setTotalResults] = useState(0);
    const [currentResultIndex, setCurrentResultIndex] = useState(0);

    const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);


    const [activeModel, setActiveModel] = useState('deepseek');
    const [explanations, setExplanations] = useState<Record<string, ExplanationData>>({});
    const [shapData, setShapData] = useState<ShapData>({});
    const [ratings, setRatings] = useState<Record<string, Record<string, number>>>({});
    const [shapRating, setShapRating] = useState(0);
    const [isSubmittingRatings, setIsSubmittingRatings] = useState(false);

    useEffect(() => {
    const fetchData = async () => {
        try {
            setLoading(true);

            // Fetch classification data
            const [entryResponse, classificationResponse] = await Promise.all([
                axios.get(`http://localhost:5000/api/classificationentry/${classificationId}/${resultId}`, { withCredentials: true }),
                axios.get(`http://localhost:5000/api/classification/${classificationId}`, { withCredentials: true })
            ]);

            const entryData = entryResponse.data;
            setClassification(entryData);
            setTotalResults(classificationResponse.data.results?.length || 0);
            setCurrentResultIndex(Number(resultId) || 0);

            // Get saved explanation models or use default if none exist
            const savedModels = classificationResponse.data.explanation_models || [
                { provider: 'deepseek', model: 'deepseek' },
                { provider: 'openai', model: 'chatgpt' },
                { provider: 'mistral', model: 'mistral' }
            ];
            console.log(entryData);

            // Initialize explanations and ratings for saved models
            const initialData: Record<string, ExplanationData> = {};
            const initialRatings: Record<string, Record<string, number>> = {};

            savedModels.forEach((model: { provider: any; model: any; }) => {
                const modelId = `${model.provider}-${model.model}`.toLowerCase();

                // Load existing explanations if they exist
                initialData[modelId] = {
                    llm: entryData.llm_explanations?.[model.model],
                    combined: entryData.shapwithllm_explanations?.[model.model]
                };

                initialRatings[modelId] = {
                    llm: 0,
                    combined: 0
                };
            });

            const modelInfos = savedModels.map((model: { provider: any; model: any; }) => ({
                id: `${model.provider}-${model.model}`.toLowerCase(),
                model: model.model,
                provider: model.provider
            }));

            setAvailableModels(modelInfos);
            setExplanations(initialData);
            setRatings(initialRatings);
            setActiveModel(Object.keys(initialData)[0] || '');

            // Load SHAP explanation if it exists
            if (entryData.shap_plot_explanation) {
                setShapData({
                    explanation: entryData.shap_plot_explanation
                });
            }

        } catch (err) {
            setError("Failed to load data");
        } finally {
            setLoading(false);
        }
    };

    fetchData();
}, [classificationId, resultId]);

    const generateShapExplanation = async () => {
        setIsExplaining(true);
        try {
            const shapResponse = await axios.post('http://localhost:5000/api/explain', {
                text: classification?.text,
                explainer_type: 'shap',
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                truelabel: classification?.actualLabel,
                classificationId:classificationId,
                resultId:resultId,

            }, { withCredentials: true });

            setShapData({
                explanation: shapResponse.data.explanation,
                shapWords: shapResponse.data.top_words
            });
            console.log(shapResponse.data.top_words,'shaping ');


        } catch (err) {
            setError('Failed to generate SHAP explanation');
        } finally {
            setIsExplaining(false);
        }
    };

    const generateLLMExplanation = async (modelId: string) => {
        setIsExplaining(true);
        const model = availableModels.find(m => m.id === modelId);
        if (!model) {
        console.error(`Model with ID "${modelId}" not found in availableModels.`);
        return;
        }

        setIsExplaining(true);
        console.log(model, '→ sending to backend');

        try {
            const llmResponse = await axios.post('http://localhost:5000/api/explain', {
                text: classification?.text,
                provider: model.provider,
                model: model.model,
                explainer_type: 'llm',
                resultId:resultId,
                predictedlabel: classification?.prediction,
                confidence: classification?.confidence,
                truelabel: classification?.actualLabel,
                classificationId:classificationId
            }, { withCredentials: true });

            // Generate combined explanation if SHAP data exists
            let combinedExplanation: null = null;
            console.log(shapData.shapWords);
            if (shapData.shapWords) {
                const combinedResponse = await axios.post('http://localhost:5000/api/explain_withshap', {
                    text: classification?.text,
                    shapwords: shapData.shapWords,
                    provider: model?.provider,
                    model: model.model,
                    label: classification?.prediction,
                    resultId:resultId,
                    confidence: classification?.confidence,
                    classificationId:classificationId
                }, { withCredentials: true });
                combinedExplanation = combinedResponse.data;
            }

            // @ts-ignore
            setExplanations(prev => ({
                ...prev,
                [modelId]: {
                    llm: llmResponse.data.explanation,
                    combined: combinedExplanation
                }
            }));

        } catch (err) {
            setError(`Failed to generate explanations for ${modelId}`);
        } finally {
            setIsExplaining(false);
        }
    };

    const generateAllExplanations = async () => {
        setIsExplaining(true);
        try {
            // First generate SHAP if not exists
            if (!shapData.explanation) {
                await generateShapExplanation();
            }

            // Then generate LLM explanations for all models
            for (const model of availableModels) {
                await generateLLMExplanation(model.id);
            }
        } catch (err) {
            setError('Failed to generate all explanations');
        } finally {
            setIsExplaining(false);
        }
    };

    const get_faithfulness = async (modelId: string) => {
    setIsFetchingFaithfulness(true);
    setFaithfulnessError(null);
    setFaithfulnessScore(null);

    try {
        const model = availableModels.find(m => m.id === modelId);
        if (!model) {
            setFaithfulnessError("Model not found.");
            return;
        }

        const payload = {
            ground_question: classification?.text,   // or the actual ground question if different
            ground_explanation: classification?.llm_explanations?.[model.model] || "", // or other field
            ground_label: classification?.actualLabel,
            predicted_explanation: explanations[modelId]?.llm || "",
            predicted_label: classification?.prediction,
            target_model: model.model,
            // Optionally add context, groq, target_model if needed
        };

        const response = await axios.post("http://localhost:5000/api/faithfulness", payload, { withCredentials: true });

        setFaithfulnessScore(response.data.faithfulness_score);
    } catch (err: any) {
        setFaithfulnessError('Failed to compute faithfulness');
    } finally {
        setIsFetchingFaithfulness(false);
    }
};

    const handleRatingChange = (modelId: string, type: string, rating: number) => {
        setRatings(prev => ({
            ...prev,
            [modelId]: {
                ...prev[modelId],
                [type]: rating
            }
        }));
    };

    const submitRatings = async () => {
        setIsSubmittingRatings(true);
        try {
            await axios.post(
                'http://localhost:5000/api/submit-ratings',
                {
                    classificationId,
                    resultId,
                    ratings,
                    shapRating,
                    timestamp: new Date().toISOString()
                },
                { withCredentials: true }
            );
            alert('Ratings submitted successfully!');
        } catch (err) {
            setError('Failed to submit ratings');
        } finally {
            setIsSubmittingRatings(false);
        }
    };

    const hasRatings = () => {
        const hasModelRatings = Object.values(ratings).some(modelRatings =>
            Object.values(modelRatings).some(rating => rating > 0)
        );
        return hasModelRatings || shapRating > 0;
    };

    const handlePrevious = () => {
        const newIndex = currentResultIndex - 1;
        navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
    };

    const handleNext = () => {
        const newIndex = currentResultIndex + 1;
        navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
    };

    if (loading) {
        return (
            <Container className="py-5 text-center">
                <Spinner animation="border" variant="primary" />
                <p className="mt-3">Loading classification data...</p>
            </Container>
        );
    }

    if (error) {
        return (
            <Container className="py-5">
                <Alert variant="danger">{error}</Alert>
                <Button variant="secondary" onClick={() => navigate(-1)}>Go Back</Button>
            </Container>
        );
    }

    return (
        <Container className="py-4 explanation-page" fluid>
            {/* Navigation Header */}
            <div className="d-flex justify-content-between align-items-center mb-4">
                <Button
                    variant="outline-secondary"
                    onClick={() => navigate(`/datasets/${datasetId}/classifications/${classificationId}`)}
                >
                    ← Back to Classification
                </Button>

                <div className="d-flex align-items-center gap-3">
                    <div className="text-muted">
                        Result {currentResultIndex + 1} of {totalResults}
                    </div>
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
            </div>

            {/* Classification Summary */}
            {classification && (
                <Card className="mb-4">
                    <Card.Body>
                        <Row>
                            <Col md={8}>
                                <h5>Original Text</h5>
                                <div className="original-text p-3 bg-light rounded">
                                    {classification.text}
                                </div>
                                <div className="text-muted small mt-2">
                                    Confidence: {(classification.confidence * 100).toFixed(1)}%
                                </div>
                            </Col>
                            <Col md={4}>
                                <div className="d-flex flex-column gap-3">
                                    <div className="text-center">
                                        <div className="text-muted small">Prediction</div>
                                        <Badge
                                            pill
                                            bg={classification.prediction === 'POSITIVE' ? 'success' : 'danger'}
                                            className="px-3 py-2 fs-6"
                                        >
                                            {classification.prediction}
                                        </Badge>
                                    </div>
                                    <div className="text-center">
                                        <div className="text-muted small">Actual Label</div>
                                        <Badge
                                            pill
                                            bg={classification.actualLabel === 1 ? 'success' : 'danger'}
                                            className="px-3 py-2 fs-6"
                                        >
                                            {classification.actualLabel === 1 ? 'POSITIVE' : 'NEGATIVE'}
                                        </Badge>
                                    </div>
                                </div>
                            </Col>
                        </Row>
                    </Card.Body>
                </Card>
            )}

            {/* Main Layout */}
            <Row className="g-4">
                {/* Left Column - SHAP Visualization */}
                <Col lg={4}>
                    <Card className="h-100 explanation-card border-info">
                        <Card.Header className="bg-info text-white d-flex justify-content-between align-items-center">
                            <Card.Title className="mb-0">
                                SHAP Analysis
                            </Card.Title>
                            <Button
                                size="sm"
                                variant="light"
                                onClick={generateShapExplanation}
                                disabled={isExplaining}
                            >
                                {isExplaining ? (
                                    <Spinner size="sm" animation="border" />
                                ) : shapData.explanation ? 'Regenerate' : 'Generate'}
                            </Button>
                        </Card.Header>
                        <Card.Body>
                            {shapData.explanation ? (
                                <div
                                    dangerouslySetInnerHTML={{ __html: shapData.explanation }}
                                    className="shap-visualization"
                                />
                            ) : (
                                <div className="text-muted text-center py-5">
                                    <p>Click "Generate" to create SHAP visualization</p>
                                </div>
                            )}
                        </Card.Body>
                        <Card.Footer>
                            <RatingSection
                                title="SHAP Analysis"
                                value={shapRating}
                                onChange={setShapRating}
                                disabled={!shapData.explanation}
                            />
                        </Card.Footer>
                    </Card>
                </Col>

                {/* Right Column - LLM Explanations */}
                <Col lg={8}>
                    <Card className="h-100">
                        <Card.Header>
                            <div className="d-flex justify-content-between align-items-center">
                                <Card.Title className="mb-0">
                                    <span className="me-2">🤖</span>
                                    LLM Explanations
                                </Card.Title>
                                <div className="d-flex gap-2">
                                    <Button
                                        size="sm"
                                        variant="outline-primary"
                                        onClick={() => generateLLMExplanation(activeModel)}
                                        disabled={isExplaining}
                                    >
                                        Generate Current
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="primary"
                                        onClick={generateAllExplanations}
                                        disabled={isExplaining}
                                    >
                                        {isExplaining ? (
                                            <Spinner size="sm" className="me-2" />
                                        ) : null}
                                        Generate All
                                    </Button>
                                </div>
                            </div>
                        </Card.Header>
                        <Card.Body className="p-0">
                            {/* Model Selection Tabs */}
                            <Tabs
                                activeKey={activeModel}
                                onSelect={(k) => setActiveModel(k as string)}
                                className="model-tabs border-bottom-0"
                                fill
                            >
                                {availableModels.map(model => (
                                    <Tab
                                        key={model.id}
                                        eventKey={model.id}
                                        title={
                                            <div className="d-flex align-items-center justify-content-center gap-2">
                                                {model.model}
                                            </div>
                                        }
                                    >
                                        <div className="p-4">
                                            <Row className="g-4">
                                                {/* LLM Explanation */}
                                                <Col md={6}>
                                                    <div className="explanation-section">
                                                        <h6 className="text-primary mb-3">Direct Explanation</h6>
                                                        <div className="explanation-content mb-3">
                                                            {explanations[activeModel]?.llm ? (
                                                                <div className="p-3 bg-light rounded">
                                                                    {explanations[activeModel].llm}
                                                                </div>
                                                            ) : (
                                                                <div className="text-muted text-center py-4 border rounded">
                                                                    No explanation generated yet
                                                                </div>
                                                            )}
                                                        </div>
                                                        {/* Faithfulness button and score */}
                                                        <div className="d-flex align-items-center gap-3 my-3">
                                                          <Button
                                                            size="sm"
                                                            variant="outline-info"
                                                            onClick={() => get_faithfulness(activeModel)}
                                                            disabled={isFetchingFaithfulness || !explanations[activeModel]?.llm}
                                                          >
                                                            {isFetchingFaithfulness ? <Spinner size="sm" /> : "Compute Faithfulness"}
                                                          </Button>

                                                          {faithfulnessScore !== null && (
                                                            <div className="d-flex align-items-center">
                                                              <span className="badge rounded-pill bg-info fs-6 px-3 py-2">
                                                                Faithfulness: {faithfulnessScore.toFixed(2)}
                                                              </span>
                                                            </div>
                                                          )}

                                                          {faithfulnessError && (
                                                            <span className="text-danger ms-2">{faithfulnessError}</span>
                                                          )}
                                                        </div>
                                                        <RatingSection
                                                            title="Direct Explanation"
                                                            value={ratings[activeModel]?.llm || 0}
                                                            onChange={(rating: number) => handleRatingChange(activeModel, 'llm', rating)}
                                                            disabled={!explanations[activeModel]?.llm}
                                                        />
                                                    </div>
                                                </Col>

                                                {/* Combined Analysis */}
                                                <Col md={6}>
                                                    <div className="explanation-section">
                                                        <h6 className="text-success mb-3">
                                                            SHAP-Enhanced Analysis
                                                        </h6>
                                                        <div className="explanation-content mb-3">
                                                            {explanations[activeModel]?.combined ? (
                                                                <div className="p-3 bg-light rounded">
                                                                    {explanations[activeModel].combined}
                                                                </div>
                                                            ) : (
                                                                <div className="text-muted text-center py-4 border rounded">
                                                                    {!shapData.shapWords ?
                                                                        "Generate SHAP analysis first" :
                                                                        "Generate combined analysis"
                                                                    }
                                                                </div>
                                                            )}
                                                        </div>
                                                        <RatingSection
                                                            title="Combined Analysis"
                                                            value={ratings[activeModel]?.combined || 0}
                                                            onChange={(rating: number) => handleRatingChange(activeModel, 'combined', rating)}
                                                            disabled={!explanations[activeModel]?.combined}
                                                        />
                                                    </div>
                                                </Col>
                                            </Row>
                                        </div>
                                    </Tab>
                                ))}
                            </Tabs>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            {/* Submit Ratings */}
            <div className="d-flex justify-content-end mt-4">
                <Button
                    variant="success"
                    size="lg"
                    onClick={submitRatings}
                    disabled={isSubmittingRatings || !hasRatings()}
                    className="submit-ratings-btn"
                >
                    {isSubmittingRatings ? (
                        <Spinner size="sm" className="me-2" />
                    ) : null}
                    Submit All Ratings ({Object.keys(ratings).length * 2 + 1} explanations)
                </Button>
            </div>
        </Container>
    );
};

// Rating Component
interface RatingSectionProps {
    title: string;
    value: number;
    onChange: (rating: number) => void;
    disabled: boolean;
}

const RatingSection: React.FC<RatingSectionProps> = ({ title, value, onChange, disabled }) => (
    <div className="rating-section">
        <div className="d-flex justify-content-between align-items-center">
            <span className="small text-muted">Rate {title}:</span>
            <div className="d-flex gap-1">
                {[1, 2, 3, 4, 5].map((rating) => (
                    <button
                        key={rating}
                        className={`rating-star ${value >= rating ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
                        onClick={() => !disabled && onChange(rating)}
                        disabled={disabled}
                        style={{
                            minWidth: '30px',
                            height: '30px',
                            border: '1px solid #dee2e6',
                            borderRadius: '4px',
                            backgroundColor: value >= rating ? '#007bff' : 'white',
                            color: value >= rating ? 'white' : '#6c757d',
                            cursor: disabled ? 'not-allowed' : 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                            opacity: disabled ? 0.5 : 1
                        }}
                    >
                        {rating}
                    </button>
                ))}
            </div>
        </div>
        {value > 0 && (
            <div className="text-end small mt-1">
                <span className="text-muted">Your rating:</span> {value}/5
            </div>
        )}
    </div>
);

export default ExplanationPage;