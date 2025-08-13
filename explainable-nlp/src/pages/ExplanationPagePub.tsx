import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Alert, Spinner, Button, Tab, Tabs, Badge } from 'react-bootstrap';
import axios from 'axios';
import '../index.css';

interface PubMedQAEntry {
  question: string;
  context: string;         // Abstract as plain string!
  prediction: string;      // Model's answer ("yes"/"no")
  actualLabel?: string;    // True label ("yes"/"no")
  score: number;           // Model's confidence
  method?: string;
  llm_explanations?: Record<string, string>;
  shap_plot_explanation?: string;
  shapwithllm_explanations?: Record<string, string>;
  long_answer: string;
  trustworthiness_score: number;
  plausibility_score: number;
  faithfulness_score: number;
  ratings?: Record<string, { llm?: number; combined?: number }>;
  correctness: number;
  consistency: number;
  qag_score: number;
  counterfactual: number;
  contextual_faithfulness: number;
}

interface ModelInfo {
  provider: string;
  model: string;
  id: string;
}

interface ExplanationData {
  llm?: string;
  combined?: string;
}

interface ShapData {
  explanation?: string;
  shapWords?: string[];
}

const ExplanationPagePubMedQA = () => {
  const { datasetId, classificationId, resultId } = useParams();
  const navigate = useNavigate();

  const [entry, setEntry] = useState<PubMedQAEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalResults, setTotalResults] = useState(0);
  const [currentResultIndex, setCurrentResultIndex] = useState(0);

  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [activeModel, setActiveModel] = useState<string>('');
  const [explanations, setExplanations] = useState<Record<string, ExplanationData>>({});
  const [shapData, setShapData] = useState<ShapData>({});
  const [ratings, setRatings] = useState<Record<string, Record<string, number>>>({});
  const [shapRating, setShapRating] = useState(0);

  // Faithfulness state
  const [faithfulnessScores, setFaithfulnessScores] = useState<Record<string, { llm: number | null, combined: number | null }>>({});
  const [isFetchingFaithfulness, setIsFetchingFaithfulness] = useState<{ modelId: string, type: string } | null>(null);
  const [faithfulnessError, setFaithfulnessError] = useState<string | null>(null);

  const [isExplaining, setIsExplaining] = useState(false);
  const [isSubmittingRatings, setIsSubmittingRatings] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      setShapData({});
      try {
        const [entryRes, classRes] = await Promise.all([
          axios.get(`http://localhost:5000/api/classificationentry/${classificationId}/${resultId}`, { withCredentials: true }),
          axios.get(`http://localhost:5000/api/classification/${classificationId}`, { withCredentials: true }),
        ]);
        setEntry(entryRes.data);
        console.log(entryRes.data);
        setTotalResults(classRes.data.results?.length || 0);
        setCurrentResultIndex(Number(resultId) || 0);

        // Use provided models or defaults
        const savedModels = classRes.data.explanation_models || [
          { provider: 'deepseek', model: 'deepseek' },
          { provider: 'openai', model: 'chatgpt' },
          { provider: 'mistral', model: 'mistral' }
        ];

        const initialData: Record<string, ExplanationData> = {};
        const initialRatings: Record<string, Record<string, number>> = {};
        const initialFaithfulnessScores: Record<string, { llm: number | null, combined: number | null }> = {};

        savedModels.forEach((m: any) => {
          const modelId = `${m.provider}-${m.model}`.toLowerCase();
          initialData[modelId] = {
            llm: entryRes.data.llm_explanations?.[m.model],
            combined: entryRes.data.shapwithllm_explanations?.[m.model]
          };
          initialRatings[modelId] = {
            llm: entryRes.data.ratings?.[modelId]?.llm || 0,
            combined: entryRes.data.ratings?.[modelId]?.combined || 0,
          };
          initialFaithfulnessScores[modelId] = { llm: null, combined: null };
        });

        setAvailableModels(savedModels.map((m: any) => ({
          id: `${m.provider}-${m.model}`.toLowerCase(),
          provider: m.provider,
          model: m.model
        })));
        setExplanations(initialData);
        setRatings(initialRatings);
        setFaithfulnessScores(initialFaithfulnessScores);
        setActiveModel(Object.keys(initialData)[0] || '');
        if (entryRes.data.shap_plot_explanation) {
          setShapData({ explanation: entryRes.data.shap_plot_explanation });
        }
      } catch {
        setError('Failed to load data');
      }
      setLoading(false);
    };
    fetchData();
  }, [classificationId, resultId]);

  // --- SHAP & LLM EXPLANATION HANDLERS ---
  const generateShapExplanation = async () => {
    setIsExplaining(true);
    try {
      const shapResponse = await axios.post('http://localhost:5000/api/explain', {
        text: entry?.context,
        explainer_type: 'shap',
        predictedlabel: entry?.prediction,
        confidence: entry?.score,
        truelabel: entry?.actualLabel,
        classificationId,
        resultId,
      }, { withCredentials: true });
      setShapData({
        explanation: shapResponse.data.explanation,
        shapWords: shapResponse.data.top_words,
      });
      return shapResponse.data.top_words;
    } catch {
      setError('Failed to generate SHAP explanation');
      return null;
    } finally {
      setIsExplaining(false);
    }
  };

  const generateLLMExplanation = async (modelId: string, shapWords?: string[]) => {
    setIsExplaining(true);
    const model = availableModels.find(m => m.id === modelId);
    if (!model) return setIsExplaining(false);

    try {
      const llmResponse = await axios.post('http://localhost:5000/api/explain', {
        text: entry?.context,
        provider: model.provider,
        model: model.model,
        explainer_type: 'llm',
        resultId,
        predictedlabel: entry?.prediction,
        confidence: entry?.score,
        truelabel: entry?.actualLabel,
        classificationId,
        datatype: "pubmedqa"
      }, { withCredentials: true });

      let combinedExplanation: null = null;
      if (shapWords && shapWords.length > 0) {
        const combinedRes = await axios.post('http://localhost:5000/api/explain_withshap', {
          text: entry?.context,
          shapwords: shapWords,
          provider: model.provider,
          model: model.model,
          label: entry?.prediction,
          resultId,
          confidence: entry?.score,
          classificationId
        }, { withCredentials: true });
        combinedExplanation = combinedRes.data;
      }
      // @ts-ignore
      setExplanations(prev => ({
        ...prev,
        [modelId]: {
          llm: llmResponse.data.explanation,
          combined: combinedExplanation
        }
      }));

      // Reset faithfulness scores when generating new explanations
      setFaithfulnessScores(prev => ({
        ...prev,
        [modelId]: {
          llm: null,
          combined: null
        }
      }));
    } catch {
      setError('Failed to generate explanations');
    } finally {
      setIsExplaining(false);
    }
  };

  const generateAllExplanations = async () => {
    setIsExplaining(true);
    try {
      let shapWords = shapData.shapWords;
      if (!shapData.explanation) {
        shapWords = await generateShapExplanation();
        if (!shapWords) throw new Error('Failed to get SHAP words for explanation');
      }
      for (const model of availableModels) {
        await generateLLMExplanation(model.id, shapWords);
      }
    } catch {
      setError('Failed to generate all explanations');
    } finally {
      setIsExplaining(false);
    }
  };

  // --- FAITHFULNESS ---
  const get_faithfulness = async (modelId: string, type: 'llm' | 'combined') => {
    setIsFetchingFaithfulness({ modelId, type });
    setFaithfulnessError(null);

    try {
      const model = availableModels.find(m => m.id === modelId);
      if (!model) {
        setFaithfulnessError("Model not found.");
        return;
      }

      const explanationToEvaluate = type === 'llm'
        ? explanations[modelId]?.llm || ""
        : explanations[modelId]?.combined || "";

      const payload = {
        ground_question: entry?.question,
        ground_explanation: entry?.context,
        ground_label: entry?.actualLabel,
        predicted_explanation: explanationToEvaluate,
        predicted_label: entry?.prediction,
        target_model: model.model,
        ground_context: entry?.context
      };

      const response = await axios.post("http://localhost:5000/api/faithfulness", payload, { withCredentials: true });

      setFaithfulnessScores(prev => ({
        ...prev,
        [modelId]: {
          ...(prev[modelId] || { llm: null, combined: null }),
          [type]: response.data.faithfulness_score
        }
      }));
    } catch (err: any) {
      setFaithfulnessError(`Failed to compute ${type} faithfulness`);
    } finally {
      setIsFetchingFaithfulness(null);
    }
  };
  const get_Lext = async (modelId: string, type: 'llm' | 'combined') => {
    setIsFetchingFaithfulness({ modelId, type });
    setFaithfulnessError(null);

    try {
      const model = availableModels.find(m => m.id === modelId);
      if (!model) {
        setFaithfulnessError("Model not found.");
        return;
      }

      const explanationToEvaluate = type === 'llm'
        ? explanations[modelId]?.llm || ""
        : explanations[modelId]?.combined || "";

      const payload = {
        ground_question: entry?.question,
        ground_explanation: entry?.context,
        ground_label: entry?.actualLabel,
        predicted_explanation: explanationToEvaluate,
        predicted_label: entry?.prediction,
        target_model: model.model,
        provider:model.provider,
        ground_context: entry?.context
      };

      const response = await axios.post("http://localhost:5000/api/trustworthiness", payload, { withCredentials: true });

      setFaithfulnessScores(prev => ({
        ...prev,
        [modelId]: {
          ...(prev[modelId] || { llm: null, combined: null }),
          [type]: response.data.trustworthiness_score
        }
      }));
    } catch (err: any) {
      setFaithfulnessError(`Failed to compute ${type} faithfulness`);
    } finally {
      setIsFetchingFaithfulness(null);
    }
  };

  // --- RATINGS ---
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
        'http://localhost:5000/api/save_ratings',
        {
          classificationId,
          resultId,
          ratings,
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
    navigate(`/datasets/${datasetId}/classifications_pub/${classificationId}/results/${newIndex}`);
  };
  const handleNext = () => {
    const newIndex = currentResultIndex + 1;
    navigate(`/datasets/${datasetId}/classifications_pub/${classificationId}/results/${newIndex}`);
  };

  // ---- RENDER ----
  if (loading) return (
    <Container className="py-5 text-center">
      <Spinner animation="border" />
      <div className="mt-3">Loading PubMedQA entry...</div>
    </Container>
  );
  if (error) return (
    <Container className="py-5">
      <Alert variant="danger">{error}</Alert>
      <Button onClick={() => navigate(-1)}>Back</Button>
    </Container>
  );
  if (!entry) return null;

  // @ts-ignore
  // @ts-ignore
  return (
    <Container className="py-4 explanation-page" fluid>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <Button variant="outline-secondary" onClick={() => navigate(`/datasets/${datasetId}/classificationsp/${classificationId}`)}>
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
            >← Previous</Button>
            <Button
              variant="outline-primary"
              onClick={handleNext}
              disabled={currentResultIndex >= totalResults - 1}
            >Next →</Button>
          </div>
        </div>
      </div>

      {/* --- MAIN CARD: Question, Context, Prediction, Actual --- */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={8}>
              <h5>Question</h5>
              <div className="p-3 bg-light rounded mb-2">{entry.question}</div>
              <h6>Context (Abstract)</h6>
              <div className="original-text p-3 bg-light rounded" style={{ whiteSpace: 'pre-line' }}>
                {entry.context}
              </div>
              <div className="text-muted small mt-2">
                Confidence: {(entry.score * 100).toFixed(1)}%
              </div>
            </Col>
            <Col md={4}>
              <div className="d-flex flex-column gap-3">
                <div className="text-center">
                  <div className="text-muted small">Prediction</div>
                  <Badge pill bg={entry.prediction === "yes" ? "success" : "danger"} className="px-3 py-2 fs-6">
                    {entry.prediction}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-muted small">Actual Label</div>
                  <Badge pill bg={entry.actualLabel === "yes" ? "success" : "danger"} className="px-3 py-2 fs-6">
                    {entry.actualLabel || "N/A"}
                  </Badge>
                </div>
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      <Row className="g-4">
        {/* SHAP Analysis */}
        {(entry?.method !== 'llm' && entry?.method !== 'explore') && (
          <Col lg={4}>
            <Card className="h-100 explanation-card border-info">
              <Card.Header className="bg-info text-white d-flex justify-content-between align-items-center">
                <Card.Title className="mb-0">SHAP Analysis</Card.Title>
                <Button size="sm" variant="light" onClick={generateShapExplanation} disabled={isExplaining}>
                  {isExplaining ? (<Spinner size="sm" animation="border" />) : shapData.explanation ? 'Regenerate' : 'Generate'}
                </Button>
              </Card.Header>
              <Card.Body>
                {shapData.explanation ? (
                  <div dangerouslySetInnerHTML={{ __html: shapData.explanation }} className="shap-visualization" />
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
        )}

        {/* LLM/SHAP-Enhanced Explanations */}
        <Col lg={entry?.method === 'llm' || entry?.method === 'explore' ? 12 : 8}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex justify-content-between align-items-center">
                <Card.Title className="mb-0">
                 LLM Explanations
                </Card.Title>
                <div className="d-flex gap-2">
                  {/* Always show the buttons, even for method="llm" */}
                  <Button size="sm" variant="outline-primary" onClick={() => generateLLMExplanation(activeModel)} disabled={isExplaining}>
                    Generate Current
                  </Button>
                  {entry?.method !== 'llm' && (
                    <Button size="sm" variant="primary" onClick={generateAllExplanations} disabled={isExplaining}>
                      {isExplaining ? (<Spinner size="sm" className="me-2" />) : null} Generate All
                    </Button>
                  )}
                </div>
              </div>
            </Card.Header>
            <Card.Body className="p-0">
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
                    title={<div className="d-flex align-items-center justify-content-center gap-2">{model.model}</div>}
                  >
                    <div className="p-4">
                      <Row className="g-4">
                        <Col md={entry?.method === 'llm' || entry?.method === 'explore' ? 12 : 6}>
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
                            {/* LExT + rating for direct */}
                            {entry.trustworthiness_score !== undefined && entry.trustworthiness_score !== null ? (
                              <>
                                <div className="d-flex align-items-center gap-2 my-3 flex-wrap">
                                  <span className="badge rounded-pill bg-warning fs-6 px-3 py-2">
                                    LExT: {Number(entry.trustworthiness_score).toFixed(2)}
                                  </span>
                                  {entry.plausibility_score !== undefined && entry.plausibility_score !== null && (
                                    <span className="badge rounded-pill bg-info fs-6 px-3 py-2">
                                      Plausibility: {Number(entry.plausibility_score).toFixed(2)}
                                    </span>
                                  )}
                                  {entry.faithfulness_score !== undefined && entry.faithfulness_score !== null && (
                                    <span className="badge rounded-pill bg-secondary fs-6 px-3 py-2">
                                      Faithfulness: {Number(entry.faithfulness_score).toFixed(2)}
                                    </span>
                                  )}
                                </div>

                                {/* Sub-metrics under Plausibility */}
                                {(entry.correctness !== undefined || entry.consistency !== undefined) && (
                                  <div className="mt-1 ms-1 small text-muted d-flex flex-wrap gap-3">
                                    {entry.correctness !== undefined && entry.correctness !== null && (
                                      <span>Correctness: {Number(entry.correctness).toFixed(2)}</span>
                                    )}
                                    {entry.consistency !== undefined && entry.consistency !== null && (
                                      <span>Consistency: {Number(entry.consistency).toFixed(2)}</span>
                                    )}
                                  </div>
                                )}

                                {/* Sub-metrics under Faithfulness */}
                                {(entry.qag_score !== undefined || entry.counterfactual !== undefined || entry.contextual_faithfulness !== undefined) && (
                                  <div className="mt-1 ms-1 small text-muted d-flex flex-wrap gap-3">
                                    {entry.qag_score !== undefined && entry.qag_score !== null && (
                                      <span>QAG: {Number(entry.qag_score).toFixed(2)}</span>
                                    )}
                                    {entry.counterfactual !== undefined && entry.counterfactual !== null && (
                                      <span>Counterfactual: {Number(entry.counterfactual).toFixed(2)}</span>
                                    )}
                                    {entry.contextual_faithfulness !== undefined && entry.contextual_faithfulness !== null && (
                                      <span>Contextual: {Number(entry.contextual_faithfulness).toFixed(2)}</span>
                                    )}
                                  </div>
                                )}
                              </>
                            ) : (
                              <div className="d-flex align-items-center gap-3 my-3">
                                <Button
                                  size="sm"
                                  variant="outline-warning"
                                  onClick={() => get_Lext(activeModel, 'llm')}
                                  disabled={
                                    (isFetchingFaithfulness?.modelId === activeModel &&
                                      isFetchingFaithfulness?.type === 'llm') ||
                                    !explanations[activeModel]?.llm
                                  }
                                >
                                  {(isFetchingFaithfulness?.modelId === activeModel &&
                                    isFetchingFaithfulness?.type === 'llm') ? (
                                    <Spinner size="sm" />
                                  ) : "Compute LExT"}
                                </Button>
                                {faithfulnessScores[activeModel]?.llm !== null && (
                                  <div className="d-flex align-items-center">
                                    <span className="badge rounded-pill bg-warning fs-6 px-3 py-2">
                                      LExT: {faithfulnessScores[activeModel]?.llm?.toFixed(2)}
                                    </span>
                                  </div>
                                )}
                                {faithfulnessError &&
                                  isFetchingFaithfulness?.modelId === activeModel &&
                                  isFetchingFaithfulness?.type === 'llm' && (
                                    <span className="text-danger ms-2">{faithfulnessError}</span>
                                  )}
                              </div>
                            )}
                            <RatingSection
                              title="Direct Explanation"
                              value={ratings[activeModel]?.llm || 0}
                              onChange={(rating: number) => handleRatingChange(activeModel, 'llm', rating)}
                              disabled={!explanations[activeModel]?.llm}
                            />

                          </div>
                        </Col>
                        {(entry?.method !== 'llm' && entry?.method !== 'explore') && (
                          <Col md={6}>
                            <div className="explanation-section">
                              <h6 className="text-success mb-3">SHAP-Enhanced Analysis</h6>
                              <div className="explanation-content mb-3">
                                {explanations[activeModel]?.combined ? (
                                  <div className="p-3 bg-light rounded">
                                    {explanations[activeModel].combined}
                                  </div>
                                ) : (
                                  <div className="text-muted text-center py-4 border rounded">
                                    {!shapData.shapWords
                                      ? "Generate SHAP analysis first"
                                      : "Generate combined analysis"
                                    }
                                  </div>
                                )}
                              </div>
                              {/* LExT + rating for SHAP-enhanced */}
                              <div className="d-flex align-items-center gap-3 my-3">
                                <Button
                                  size="sm"
                                  variant="outline-warning"
                                  onClick={() => get_Lext(activeModel, 'combined')}
                                  disabled={
                                    (isFetchingFaithfulness?.modelId === activeModel &&
                                      isFetchingFaithfulness?.type === 'combined') ||
                                    !explanations[activeModel]?.combined
                                  }
                                >
                                  {(isFetchingFaithfulness?.modelId === activeModel &&
                                    isFetchingFaithfulness?.type === 'combined') ? (
                                    <Spinner size="sm" />
                                  ) : "Compute LExT"}
                                </Button>
                                {faithfulnessScores[activeModel]?.combined !== null && (
                                  <div className="d-flex align-items-center">
                                    <span className="badge rounded-pill bg-warning fs-6 px-3 py-2">
                                      LExT: {faithfulnessScores[activeModel]?.combined?.toFixed(2)}
                                    </span>
                                  </div>
                                )}
                                {faithfulnessError &&
                                  isFetchingFaithfulness?.modelId === activeModel &&
                                  isFetchingFaithfulness?.type === 'combined' && (
                                    <span className="text-danger ms-2">{faithfulnessError}</span>
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
                        )}
                      </Row>
                    </div>
                  </Tab>
                ))}
              </Tabs>
            </Card.Body>
          </Card>
        </Col>
      </Row>

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
          Submit All Ratings)
        </Button>
      </div>
    </Container>
  );
};

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

export default ExplanationPagePubMedQA;