// Copyright (c) Microsoft. All rights reserved.

using System.Threading;
using System.Threading.Tasks;
using Azure.AI.OpenAI;
using Microsoft.SemanticKernel.AI.TextCompletion;
using Microsoft.SemanticKernel.Orchestration;

namespace Microsoft.SemanticKernel.Connectors.AI.OpenAI.AzureSdk;

internal sealed class TextResult : ITextResult
{
    private readonly Choice _choice;

    public TextResult(Completions resultData, Choice choice, string modelId)
    {
        this.ModelResult = new(new TextModelResult(resultData, choice));
        this._choice = choice;
        this.ModelId = modelId;
    }

    public ModelResult ModelResult { get; }

    public string ModelId { get; }

    public Task<string> GetCompletionAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(this._choice.Text);
    }
}
