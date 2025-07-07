import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Alert, Spinner, Button, Tab, Tabs, Badge } from 'react-bootstrap';
import axios from 'axios';
import '../index.css';

interface PubMedQAEntry {
  question: string;
  context: string;         // Abstract as plain string!
  prediction: string;           // Model's answer ("yes"/"no")
  actualLabel?: string;    // True label ("yes"/"no")
  score: number;           // Model's confidence
  method?: string;
  llm_explanations?: Record<string, string>;
  shap_plot_explanation?: string;
  shapwithllm_explanations?: Record<string, string>;
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
  const [isExplaining, setIsExplaining] = useState(false);

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
        console.log(entryRes);
        setEntry(entryRes.data);
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
        savedModels.forEach((m: any) => {
          const modelId = `${m.provider}-${m.model}`.toLowerCase();
          initialData[modelId] = {
            llm: entryRes.data.llm_explanations?.[m.model],
            combined: entryRes.data.shapwithllm_explanations?.[m.model]
          };
          initialRatings[modelId] = { llm: 0, combined: 0 };
        });
        setAvailableModels(savedModels.map((m: any) => ({
          id: `${m.provider}-${m.model}`.toLowerCase(),
          provider: m.provider,
          model: m.model
        })));
        setExplanations(initialData);
        setRatings(initialRatings);
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

  // --- SHAP & LLM EXPLANATION HANDLERS: identical to your base version! ---
  const generateShapExplanation = async () => {
    setIsExplaining(true);
    try {
      const shapResponse = await axios.post('http://localhost:5000/api/explain', {
        text: entry?.context,
        explainer_type: 'shap',
        predictedlabel: entry?.prediction,
        confidence: entry?.score,
        truelabel: entry?.actualLabel,
        classificationId: classificationId,
        resultId: resultId,
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

  // ---- Navigation etc. ----
  const handlePrevious = () => {
    const newIndex = currentResultIndex - 1;
    navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
  };
  const handleNext = () => {
    const newIndex = currentResultIndex + 1;
    navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${newIndex}`);
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
                    {entry.prediction?.toUpperCase()}
                  </Badge>
                </div>
                <div className="text-center">
                  <div className="text-muted small">Actual Label</div>
                  <Badge pill bg={entry.actualLabel === "yes" ? "success" : "danger"} className="px-3 py-2 fs-6">
                    {entry.actualLabel ? entry.actualLabel.toUpperCase() : "N/A"}
                  </Badge>
                </div>
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      <Row className="g-4">
        {/* SHAP Analysis */}
        {entry?.method !== 'llm' && (
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
              {/* Optionally: SHAP rating section */}
            </Card>
          </Col>
        )}

        {/* LLM/SHAP-Enhanced Explanations */}
        <Col lg={entry?.method === 'llm' ? 12 : 8}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex justify-content-between align-items-center">
                <Card.Title className="mb-0">
                  <span className="me-2">🤖</span> LLM Explanations
                </Card.Title>
                {entry?.method !== 'llm' && (
                  <div className="d-flex gap-2">
                    <Button size="sm" variant="outline-primary" onClick={() => generateLLMExplanation(activeModel)} disabled={isExplaining}>Generate Current</Button>
                    <Button size="sm" variant="primary" onClick={generateAllExplanations} disabled={isExplaining}>
                      {isExplaining ? (<Spinner size="sm" className="me-2" />) : null} Generate All
                    </Button>
                  </div>
                )}
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
                        <Col md={entry?.method === 'llm' ? 12 : 6}>
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
                          </div>
                        </Col>
                        {entry?.method !== 'llm' && (
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
    </Container>
  );
};

export default ExplanationPagePubMedQA;