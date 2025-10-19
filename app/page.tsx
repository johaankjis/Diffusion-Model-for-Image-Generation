"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Activity, Zap, Shield, CheckCircle2, Play, Download, TrendingDown } from "lucide-react"

export default function DiffusionModelApp() {
  const [inferenceSteps, setInferenceSteps] = useState([50])
  const [guidanceScale, setGuidanceScale] = useState([7.5])
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedImages, setGeneratedImages] = useState<string[]>([])

  const handleGenerate = async () => {
    setIsGenerating(true)
    // Simulate generation
    setTimeout(() => {
      setGeneratedImages([
        "/abstract-digital-composition.png",
        "/landscape-painting.png",
        "/portrait-art.png",
        "/futuristic-cityscape.png",
      ])
      setIsGenerating(false)
    }, 2000)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <Activity className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-foreground">DDPM Pipeline</h1>
                <p className="text-sm text-muted-foreground">Diffusion Model Research Platform</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="gap-1.5">
                <div className="h-2 w-2 rounded-full bg-green-500" />
                Model Ready
              </Badge>
              <Button variant="outline" size="sm">
                <Download className="mr-2 h-4 w-4" />
                Export
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Stats Overview */}
        <div className="mb-8 grid gap-6 md:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">FID Score</CardTitle>
              <TrendingDown className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">12.4</div>
              <p className="text-xs text-green-500">18% improvement</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Inference Time</CardTitle>
              <Zap className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">2.1s</div>
              <p className="text-xs text-blue-500">30% faster</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Privacy Status</CardTitle>
              <Shield className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">100%</div>
              <p className="text-xs text-purple-500">PII Protected</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Reproducibility</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">Verified</div>
              <p className="text-xs text-muted-foreground">All artifacts ready</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="inference" className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="inference">Inference</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
          </TabsList>

          {/* Inference Tab */}
          <TabsContent value="inference" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-3">
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle>Generation Controls</CardTitle>
                  <CardDescription>Configure sampling parameters</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-3">
                    <Label htmlFor="prompt">Prompt (Optional)</Label>
                    <Input id="prompt" placeholder="Enter generation prompt..." />
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label>Inference Steps</Label>
                      <span className="text-sm text-muted-foreground">{inferenceSteps[0]}</span>
                    </div>
                    <Slider value={inferenceSteps} onValueChange={setInferenceSteps} min={10} max={100} step={10} />
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label>Guidance Scale</Label>
                      <span className="text-sm text-muted-foreground">{guidanceScale[0]}</span>
                    </div>
                    <Slider value={guidanceScale} onValueChange={setGuidanceScale} min={1} max={15} step={0.5} />
                  </div>

                  <Button className="w-full" onClick={handleGenerate} disabled={isGenerating}>
                    {isGenerating ? (
                      <>Generating...</>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Generate Images
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Generated Results</CardTitle>
                  <CardDescription>High-quality samples from DDPM pipeline</CardDescription>
                </CardHeader>
                <CardContent>
                  {isGenerating ? (
                    <div className="flex h-96 items-center justify-center">
                      <div className="text-center">
                        <div className="mb-4 inline-block h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                        <p className="text-sm text-muted-foreground">Generating images...</p>
                      </div>
                    </div>
                  ) : generatedImages.length > 0 ? (
                    <div className="grid grid-cols-2 gap-4">
                      {generatedImages.map((img, idx) => (
                        <div key={idx} className="group relative overflow-hidden rounded-lg border border-border">
                          <img
                            src={img || "/placeholder.svg"}
                            alt={`Generated ${idx + 1}`}
                            className="h-full w-full object-cover"
                          />
                          <div className="absolute inset-0 flex items-center justify-center bg-black/60 opacity-0 transition-opacity group-hover:opacity-100">
                            <Button size="sm" variant="secondary">
                              <Download className="mr-2 h-4 w-4" />
                              Download
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex h-96 items-center justify-center text-center">
                      <div>
                        <Activity className="mx-auto mb-4 h-12 w-12 text-muted-foreground" />
                        <p className="text-sm text-muted-foreground">No images generated yet</p>
                        <p className="text-xs text-muted-foreground">Configure parameters and click Generate</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Training Tab */}
          <TabsContent value="training" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Training Progress</CardTitle>
                <CardDescription>Monitor DDPM training metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-foreground">Epoch 45/100</span>
                    <span className="text-sm text-muted-foreground">45%</span>
                  </div>
                  <Progress value={45} />
                </div>

                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Training Loss</p>
                    <p className="text-2xl font-bold text-foreground">0.0234</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Learning Rate</p>
                    <p className="text-2xl font-bold text-foreground">1e-4</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Time Remaining</p>
                    <p className="text-2xl font-bold text-foreground">2.5h</p>
                  </div>
                </div>

                <div className="rounded-lg border border-border bg-muted/50 p-4">
                  <h4 className="mb-3 text-sm font-medium text-foreground">Recent Checkpoints</h4>
                  <div className="space-y-2">
                    {["checkpoint_epoch_45.pt", "checkpoint_epoch_40.pt", "checkpoint_epoch_35.pt"].map(
                      (checkpoint, idx) => (
                        <div key={idx} className="flex items-center justify-between rounded-md bg-background p-3">
                          <span className="text-sm text-foreground">{checkpoint}</span>
                          <Button size="sm" variant="ghost">
                            Download
                          </Button>
                        </div>
                      ),
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Evaluation Tab */}
          <TabsContent value="evaluation" className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Quality Metrics</CardTitle>
                  <CardDescription>Quantitative evaluation results</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">FID Score</span>
                      <span className="text-sm font-medium text-foreground">12.4</span>
                    </div>
                    <Progress value={82} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Inception Score</span>
                      <span className="text-sm font-medium text-foreground">8.7</span>
                    </div>
                    <Progress value={87} className="h-2" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">LPIPS Distance</span>
                      <span className="text-sm font-medium text-foreground">0.15</span>
                    </div>
                    <Progress value={75} className="h-2" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Compliance Status</CardTitle>
                  <CardDescription>Privacy and fairness safeguards</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                      <span className="text-sm text-foreground">PII Redaction</span>
                    </div>
                    <Badge variant="outline" className="bg-green-500/10 text-green-500">
                      Active
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                      <span className="text-sm text-foreground">Data Anonymization</span>
                    </div>
                    <Badge variant="outline" className="bg-green-500/10 text-green-500">
                      Active
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                      <span className="text-sm text-foreground">Fairness Validation</span>
                    </div>
                    <Badge variant="outline" className="bg-green-500/10 text-green-500">
                      Passed
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
