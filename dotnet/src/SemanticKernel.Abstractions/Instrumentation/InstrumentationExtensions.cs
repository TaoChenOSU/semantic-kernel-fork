// Copyright (c) Microsoft. All rights reserved.

using System.Diagnostics;

namespace Microsoft.SemanticKernel.Instrumentation;

/// <summary>
/// Provides a set of extension methods for instrumentation related types.
/// </summary>
public static class InstrumentationExtensions
{
    /// <summary>
    /// Adds a list of tags to the activity.
    /// </summary>
    /// <param name="activity">The activity to add the tags to.</param>
    /// <param name="tagList">The tags to add.</param>
    public static void AddTags(this Activity activity, TagList tagList)
    {
        foreach (var tag in tagList)
        {
            activity.AddTag(tag.Key, tag.Value);
        }
    }
}
