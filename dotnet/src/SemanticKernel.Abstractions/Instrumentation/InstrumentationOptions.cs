// Copyright (c) Microsoft. All rights reserved.

namespace Microsoft.SemanticKernel.Instrumentation;

/// <summary>
/// Options for instrumentation.
/// </summary>
public class InstrumentationOptions
{
    /// <summary>
    /// Controls whether sensitive data should be included in the instrumentation.
    /// Sensitive data includes the following:
    /// - Function inputs and outputs
    /// - Planner goals and steps
    /// </summary>
    public bool IncludeSensitiveData { get; set; } = false;
}